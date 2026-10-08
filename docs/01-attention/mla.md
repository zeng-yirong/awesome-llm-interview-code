# 多头潜在注意力 (Multi-Latent Attention, MLA)

> KV 低秩压缩到潜空间，DeepSeek-V2 核心

## 📌 原理与思想

将 KV 先下投影压缩到低维潜空间（存入 Cache），再上投影恢复。压缩比可达 90%+，远超 GQA。

**为什么需要 MLA？**
- GQA: KV Cache 减少到 25%（8/32）
- MLA: KV Cache 减少到 <10%
- 通过低秩压缩实现极致效率

## 📐 核心公式

```
KV 压缩: c_kv = W_down(x)  → [B, S, latent_dim]  (存入 Cache)
KV 恢复: K,V = W_up(c_kv)  → [B, S, H, (Dh+Dr+Dh)]

Q 压缩: c_q = W_down_q(x) → W_up_q → [B, S, H, (Dh+Dr)]

对比:
- GQA: KV Cache = 2 × G × Dh
- MLA: KV Cache = latent_dim (远小于 GQA)
```

## 📊 张量流程图

```
x: [B, S, D]
  │
  ├── kv_down → [B, S, C]  ←── 只缓存这个! (压缩后)
  │       │
  │    kv_up → split → k_content, k_rope, v
  │                                │
  ├── q_down → [B, S, C]         │
  │       │                       │
  │    q_up → split → q_content, q_rope
  │                                │
  │    RoPE(q_rope, k_rope) ──────┘
  │            │
  │    q = cat(q_content, q_rope)
  │    k = cat(k_content, k_rope)
  │            │
  └── SDPA(q, k, v) → Wo → output
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLA(nn.Module):
    def __init__(self, d_model, n_heads, dh, latent_dim, rope_dim):
        super().__init__()
        self.h, self.dh, self.dr = n_heads, dh, rope_dim
        
        # KV 压缩投影
        self.kv_down = nn.Linear(d_model, latent_dim, bias=False)
        self.kv_up = nn.Linear(latent_dim, n_heads*(dh+rope_dim+dh), bias=False)
        
        # Q 压缩投影
        self.q_down = nn.Linear(d_model, latent_dim, bias=False)
        self.q_up = nn.Linear(latent_dim, n_heads*(dh+rope_dim), bias=False)
        
        self.wo = nn.Linear(n_heads*dh, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        
        # KV 路径
        kv = self.kv_up(self.kv_down(x)).view(B, S, self.h, -1)
        k_c, k_r, v = kv.split([self.dh, self.dr, self.dh], dim=-1)
        
        # Q 路径
        q = self.q_up(self.q_down(x)).view(B, S, self.h, -1)
        q_c, q_r = q.split([self.dh, self.dr], dim=-1)
        
        # RoPE (需要配合 RotaryEmbedding)
        # q_r, k_r = rope(q_r, k_r)
        
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh+self.dr)
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    mla = MLA(D, n_heads=8, dh=64, latent_dim=128, rope_dim=32)
    out = mla(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 512]
    print(f"Output shape: {out.shape}") # [2, 10, 512]
```

## 🎯 面试要点

- **KV 先压缩到 latent_dim 再恢复**，Cache 只存压缩后向量
- **压缩比可达 90%+** (vs GQA 75%)
- **Q 也使用低秩投影**，但不缓存（只用于当前 token）
- **DeepSeek-V2 用 MLA + MoE** 实现极高效率
- **面试重点**: 理解低秩压缩的思想，不要求完整实现


