# 多头潜在注意力 (Multi-Latent Attention, MLA)

> KV 低秩压缩到潜空间，DeepSeek-V2 核心

## 📌 原理与思想

### 核心概念
将 KV 先下投影压缩到低维潜空间（存入 Cache），再上投影恢复。压缩比可达 90%+，远超 GQA 的 75%。相比 GQA，MLA 通过低秩压缩实现更极致的 KV Cache 压缩，是 DeepSeek-V2 的核心创新。

### 核心思想
利用低秩矩阵分解压缩 KV：先下投影到 latent_dim（存入 Cache），再上投影恢复完整的 K/V。Q 也使用低秩投影，但不缓存（只用于当前 token）。RoPE 只应用于 k_rope 和 q_rope 部分。

### 算法步骤
1. KV 下投影压缩：c_kv = W_down(x) → [B, S, latent_dim]
2. KV 上投影恢复：K,V = W_up(c_kv) → split → k_content, k_rope, v
3. Q 下投影压缩：c_q = W_down_q(x)
4. Q 上投影恢复：q = W_up_q(c_q) → split → q_content, q_rope
5. 应用 RoPE：q_rope, k_rope = rope(q_rope, k_rope)
6. 合并内容：q = cat(q_content, q_rope), k = cat(k_content, k_rope)
7. 计算注意力并输出投影

## 📐 核心公式

$$
c^{KV}=x\,W^{DKV}\in\mathbb{R}^{B\times S\times C},\qquad
[K;V]=\text{split}\!\left(c^{KV}W^{UK}\right)
$$

$$
c^{Q}=x\,W^{DQ},\qquad
Q=\text{split}\!\left(c^{Q}W^{UQ}\right)
$$

$$
k=[k_{\text{content}};k_{\text{rope}}],\qquad q=[q_{\text{content}};q_{\text{rope}}]
$$

$$
\begin{aligned}
\text{GQA}: &\quad \text{KV cache}=2\,G\,D_h\\[2pt]
\text{MLA}: &\quad \text{KV cache}=C\qquad (C=512,\ D_h=128)
\end{aligned}
$$

## 📊 张量流程图

```
# 只缓存低维潜变量，用的时候再恢复
x :: [B, S, D] :: 输入
c_kv = x·W_dkv :: [B, S, C] :: 下投影到潜空间，C 远小于 H·Dh
> 进 cache 的只有 c_kv —— 压缩发生在存储维度上，不是头数上
kv = c_kv·W_ukv :: [B, S, H·(Dh+Dr+Dh)] :: 用时上投影恢复
split :: k_content, k_rope, v :: 内容分量与 RoPE 分量分开
+ q_content :: [B, S, H, Dh] :: 同样先低秩压缩再上投影
+ q_rope :: [B, S, H, Dr] :: 不压缩，单独留给 RoPE
q = cat(q_content, q_rope) :: [B, S, H, Dh+Dr] :: 拼成完整的查询
k = cat(k_content, k_rope) :: [B, S, H, Dh+Dr] :: 键同样拼接
O = SDPA(q, k, v)·W_o :: [B, S, D] :: 之后与普通注意力完全一致
$ Cache 从 2·G·Dh 降到 C：DeepSeek-V2 取 C=512，压缩比 10× 以上
> RoPE 必须单独走不压缩的分量：它是位置相关的，挤进低秩空间会被压坏
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


