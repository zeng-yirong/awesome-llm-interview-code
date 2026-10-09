# 层归一化 (Layer Normalization)

> 沿特征维度归一化，Transformer 标配

## 📌 原理与思想

在每个样本的特征维度上计算均值和方差进行归一化，再通过可学习的 gamma/beta 进行仿射变换。与 BatchNorm 不同，不依赖 batch size。

**它解决什么问题**
- 深层网络里激活值的尺度会逐层漂移，梯度要么爆炸要么消失；BatchNorm 用 batch 统计量把它压住，但统计量依赖 batch size —— batch 小、或者序列里 padding 占多数时噪声很大。
- BatchNorm 还要在推理时改用训练期攒下的滑动平均，训练与推理的行为不一致；而 NLP 的变长序列让这件事更麻烦。
- Transformer 的残差流随层数不断累加，没有归一化时深层激活会持续放大到发散。

**核心思想**
- 归一化该归一化「一个样本自己的特征」，而不是「batch 里同一位置的样本」。
- 所以只沿最后一维 `D` 求均值和方差，得到 `[B, S, 1]` 再广播回去；每个 token 独立统计，与 batch 里其他样本无关。

**算法步骤与推导**
- 沿特征维求 `μ`、`σ²` → `[B, S, 1]`，`unbiased=False` 用的是总体方差，除以 `D` 而不是 `D-1`。
- `(x - μ) / √(σ² + ε)` 把每个 token 的激活拉成均值 0、方差 1；`ε` 取 `1e-5`，只为防除零。
- 再用 `γ`、`β` 仿射回来：归一化会限制表达力（比如把激活压进线性区），让网络自己决定恢复多少尺度、多少偏移，两者都是 `[D]`。

**对比与代价**
- 对比 BatchNorm：不依赖 batch size、训练推理完全一致、天然适合变长序列；代价是丢掉了跨样本统计，也就丢了 batch 噪声带来的正则效果。
- 代价是每个 token 都要做两次归约，比后续的 RMSNorm 贵一倍；不过相对注意力那 `O(S²)` 的开销可以忽略。

## 📐 核心公式

$$
\mu=\operatorname{mean}(x,\ \dim=-1),\qquad
\sigma^{2}=\operatorname{var}(x,\ \dim=-1,\ \text{unbiased}=\text{False})
$$

$$
\operatorname{LN}(x)=\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}\odot\gamma+\beta,
\qquad
\gamma,\beta\in\mathbb{R}^{D},\qquad
\epsilon=10^{-5}
$$

## 📊 张量流程图

```
# 沿特征维归一化：每个 token 独立统计
x :: [B, S, D] :: 残差流的输入
μ = mean(x, -1, keepdim=True) :: [B, S, 1] :: 只沿最后一维求均值
σ² = var(x, -1, unbiased=False) :: [B, S, 1] :: 用总体方差，不做贝塞尔校正
x̂ = (x - μ) / √(σ² + ε) :: [B, S, D] :: 广播回原形状，每个 token 均值 0、方差 1
y = x̂·γ + β :: [B, S, D] :: γ、β 都是可学习的 [D]，负责恢复尺度与偏移
$ 统计量取自每个 token 自己的 D 维，与 batch 里其他样本无关
> 训练与推理走同一条路径，不像 BatchNorm 需要维护滑动平均
```

## 💻 代码实现

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))    # 缩放
        self.beta = nn.Parameter(torch.zeros(d_model))    # 偏移

    def forward(self, x):
        """x: [B, S, D]"""
        # 计算均值和方差（沿最后一维）
        mean = x.mean(-1, keepdim=True)
        
        # 面试陷阱: unbiased=False (除以 N, 不是 N-1)
        var = x.var(-1, keepdim=True, unbiased=False)
        
        # 归一化 + 仿射变换
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 64
    x = torch.randn(B, S, D)
    ln = LayerNorm(D)
    out = ln(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 64]
    print(f"Output shape: {out.shape}") # [2, 10, 64]
    
    # 验证归一化效果
    print(f"Mean: {out[0, 0].mean().item():.6f}")   # ≈ 0
    print(f"Var: {out[0, 0].var().item():.6f}")     # ≈ 1
```

## 🎯 面试要点

- **面试陷阱**: `torch.var` 默认 `unbiased=True` (N-1)，LN 要用 `False` (N)
- **gamma(缩放) 和 beta(偏移)** 两个可学习参数
- **Pre-Norm**: `x = x + SubLayer(LN(x))` — 现代 LLM 标配
- **Post-Norm**: `x = LN(x + SubLayer(x))` — 原始 Transformer
- **对比 BatchNorm**: BN 沿 batch 维度归一化，LN 沿特征维度


