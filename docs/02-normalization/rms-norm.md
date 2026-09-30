# RMS 归一化 (RMS Normalization)

> 去掉均值中心化，LLaMA/Mistral 标配

## 📌 原理与思想

LayerNorm 的简化版：不做均值中心化，只用 RMS (均方根) 归一化。计算更快，效果相当。

**为什么用 RMSNorm？**
- 去掉均值中心化，计算更快
- 只有一个可学习参数 gamma（无 beta）
- 实践中效果与 LayerNorm 相当

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
x: [B, S, D]
  │
  ├── x.float().pow(2).mean(-1, keepdim=True) → ms: [B,S,1]
  ├── rsqrt(ms + eps) → rsqrt: [B,S,1]
  │
  └── (x.float() * rsqrt).type_as(x) × γ → [B, S, D]
                                              γ: [D] (只有缩放, 无偏移)
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

---

**来源**: [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) + [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)
