# 标准前馈网络 (Feed-Forward Network)

> 两层 MLP，占 Transformer 2/3 参数量

## 📌 原理与思想

注意力层之后的两层全连接网络。先上投影扩展维度（通常 4 倍），应用激活函数，再下投影恢复维度。

**为什么需要 FFN？**
- Attention 捕捉 token 间的关系
- FFN 对每个 token 独立做非线性变换
- FFN 占 Transformer 参数量的 2/3

## 📐 核心公式

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂

参数量: D × 4D + 4D × D = 8D² (vs Attention: 4D²)

其中:
- W₁: [D, 4D] 上投影
- W₂: [4D, D] 下投影
- 中间维度通常是 d_model 的 4 倍
```

## 📊 张量流程图

```
x: [B, S, D]
  │
  W₁: [D, 4D]
  │
  → [B, S, 4D] → ReLU → [B, S, 4D]
                              │
  W₂: [4D, D]                 │
  │                           │
  ←───────────────────────────┘
  → [B, S, D]
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """x: [B, S, D]"""
        return self.w2(F.relu(self.w1(x)))

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    ffn = FFN(D)
    out = ffn(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 512]
    print(f"Output shape: {out.shape}") # [2, 10, 512]
    
    # 统计参数量
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"FFN params: {total_params:,}")  # 约 4M
```

## 🎯 面试要点

- **中间维度通常 4× d_model**
- **占 Transformer 参数量的 2/3**
- **现代 LLM 多用 SwiGLU 替代标准 FFN+ReLU**
- **MoE 本质上是对 FFN 层的稀疏化**
- **面试常问**: FFN 的作用是什么？为什么参数量这么大？


