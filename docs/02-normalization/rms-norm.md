# RMS 归一化 (RMS Normalization)

> 去掉均值中心化，LLaMA/Mistral 标配

## 📌 原理与思想

LayerNorm 的简化版：不做均值中心化，只用 RMS (均方根) 归一化。计算更快，效果相当。

**它解决什么问题**
- LayerNorm 每层要做两次归约（均值 + 方差）外加一次减法，而大模型有几十层，这笔开销被层数放大。
- 归一化在 GPU 上会拆成独立的 kernel，前向反向都要多走一趟显存，属于典型的「访存瓶颈」算子。
- 实践中发现均值中心化对最终效果贡献很小 —— 去掉它照样能稳定训练，那这部分代价就不必付。

**核心思想**
- 只控制激活的「尺度」，不管「中心」：除以均方根就够了。
- 少了求均值和减均值两步，也少了 `β` 参数；`γ` 只做缩放，初始化为全 1 即可。

**算法步骤与推导**
- 先升到 fp32 求 `mean(x²)`，再 `rsqrt` 一次得到缩放因子 —— 用 `rsqrt(ms + ε)` 而不是先开方再倒数，省一次逐元素运算。
- 乘回 `x` 与 `γ` 就是输出。全程不减去均值，所以没有减法也没有额外的广播。
- 升 fp32 是必须的：`x²` 在 bf16 下动态范围不够，容易溢出或下溢，算完再转回原精度。

**对比与代价**
- 对比 LayerNorm：少一次归约、少一个 `β`，实测精度相当，LLaMA、PaLM、Qwen 都用它。
- 代价是失去平移不变性：它只约束尺度不约束中心，等于赌「均值不重要」。实践中这个赌是成立的。

## 📐 核心公式

```
RMSNorm(x) = x / √(mean(x²) + ε) × γ

对比 LayerNorm:
  LN:  (x - mean) / √(var + ε) × γ + β
  RMS: x / √(mean(x²) + ε) × γ
  
  → 去掉 mean centering 和 bias
```

## 📊 张量流程图

```
# 只压尺度，不管中心
x :: [B, S, D] :: 残差流的输入
x32 = x.float() :: [B, S, D] :: 升 fp32，bf16 下 x² 动态范围不够
ms = mean(x32², -1, keepdim=True) :: [B, S, 1] :: 均方，只沿特征维
r = rsqrt(ms + ε) :: [B, S, 1] :: ε 取 1e-5，纯数值保护
y = (x32·r).type_as(x)·γ :: [B, S, D] :: 只有一个可学习参数 γ: [D]
$ 相比 LayerNorm 少了求均值与减均值，归约次数减半、参数量减半
> LLaMA、PaLM、Qwen 都把它放在每个子层之前
```

## 💻 代码实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # 只有缩放

    def _norm(self, x):
        """RMS 归一化核心计算"""
        # 计算均方根倒数 (rsqrt 比 1/sqrt 快)
        ms = x.float().pow(2).mean(-1, keepdim=True)
        return x.float() * torch.rsqrt(ms + self.eps)

    def forward(self, x):
        """x: [B, S, D]"""
        # 在 float32 下计算保证数值稳定性
        return self._norm(x).type_as(x) * self.gamma

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 64
    x = torch.randn(B, S, D)
    rms = RMSNorm(D)
    out = rms(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 64]
    print(f"Output shape: {out.shape}") # [2, 10, 64]
```

## 🎯 面试要点

- **只有 gamma 没有 beta**（无偏移参数）
- **rsqrt 比 1/sqrt 更高效**（硬件优化）
- **在 float32 下计算保证数值稳定性**
- **LLaMA, Mistral, PaLM, Gemma 等都用 RMSNorm**
- **对比 LayerNorm**: 计算更快，参数更少，效果相当


