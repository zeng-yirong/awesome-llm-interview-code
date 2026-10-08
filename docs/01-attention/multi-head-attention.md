# 多头注意力 (Multi-Head Attention)

> 并行多头 → 拼接 → 输出投影

## 📌 原理与思想

将输入投影到多个子空间，每个头独立计算注意力，最后拼接并通过线性层融合。不同头可学习不同的注意力模式。

**核心思想**：
- 多头允许模型同时关注不同位置的不同表示子空间
- 每个头学习不同的注意力模式（如语法关系、语义关系等）
- 最后通过输出投影融合所有头的信息

## 📐 核心公式

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wₒ

其中: headᵢ = Attention(Q·WᵢQ, K·WᵢK, V·WᵢV)

- h: 头数
- WᵢQ, WᵢK, WᵢV: 第 i 个头的投影矩阵
- Wₒ: 输出投影矩阵
- head_dim = d_model / num_heads
```

## 📊 张量流程图

```
x: [B, S, D]
  │
  ├── Wq ─→ Q [B,S,D] ─→ view [B,S,H,Dh] ─→ transpose [B,H,S,Dh] ─┐
  ├── Wk ─→ K [B,S,D] ─→ view [B,S,H,Dh] ─→ transpose [B,H,S,Dh] ─┤ SDPA
  └── Wv ─→ V [B,S,D] ─→ view [B,S,H,Dh] ─→ transpose [B,H,S,Dh] ─┘
                                                                      │
                                              [B,H,S,Dh] ← transpose ←┘
                                                    │
                                              view [B,S,D] → Wo → [B,S,D]
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, xq, xk=None, mask=None):
        """
        xq: [batch, seq_len_q, d_model]
        xk: [batch, seq_len_k, d_model] (None for self-attention)
        mask: [batch, 1, seq_len_q, seq_len_k]
        """
        B = xq.size(0)
        xk = xk if xk is not None else xq  # self-attention
        
        # 线性投影
        q = self.wq(xq).view(B, -1, self.h, self.dh).transpose(1, 2)
        k = self.wk(xk).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.wv(xk).view(B, -1, self.h, self.dh).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        # 加权求和
        out = torch.matmul(attn, v)  # [B, H, S, Dh]
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.dh)
        return self.wo(out)

# 测试
if __name__ == "__main__":
    B, S, D, H = 2, 10, 64, 8
    x = torch.randn(B, S, D)
    mha = MultiHeadAttention(D, H)
    out = mha(x, x)  # self-attention
    print(f"Input shape: {x.shape}")   # [2, 10, 64]
    print(f"Output shape: {out.shape}") # [2, 10, 64]
```

## 🎯 面试要点

- **分头操作**: `view(B,S,H,Dh).transpose(1,2)` — 面试必写
- **合并操作**: `transpose(1,2).contiguous().view(B,S,D)`
- **自注意力**: Q=K=V=x
- **交叉注意力**: Q=x_dec, K=V=x_enc
- **head_dim = d_model / n_heads**，通常 64 或 128
- **参数量**: 4 × d_model² (Wq, Wk, Wv, Wo)


