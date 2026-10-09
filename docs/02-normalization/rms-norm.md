# RMS 归一化 (RMS Normalization)

> 去掉均值中心化，LLaMA/Mistral 标配

## 📌 原理与思想

### 核心概念
LayerNorm 的简化版：不做均值中心化，只用 RMS (均方根) 归一化。相比 LayerNorm 计算更快、参数更少（只有 gamma 无 beta），实践中效果相当。LLaMA/Mistral/PaLM/Gemma 等主流模型都采用。

### 核心思想
去掉均值中心化步骤，只用均方根归一化。使用 rsqrt 替代 1/sqrt，计算更高效。在 float32 下计算保证数值稳定性。

### 算法步骤
1. 计算均方值：ms = mean(x², dim=-1, keepdim=True)
2. 计算均方根倒数：rsqrt = 1/√(ms + ε)
3. 归一化：x_norm = x × rsqrt
4. 缩放：output = x_norm × γ

## 📐 核心公式

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\text{mean}(x^{2})+\epsilon}}\odot\gamma,
\qquad
\epsilon=10^{-5}
$$

$$
\begin{aligned}
\text{LayerNorm}: &\quad \frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}\odot\gamma+\beta\\[2pt]
\text{RMSNorm}: &\quad \frac{x}{\sqrt{\text{mean}(x^{2})+\epsilon}}\odot\gamma
\end{aligned}
$$

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


