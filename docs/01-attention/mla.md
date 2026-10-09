# 多头潜在注意力 (Multi-Latent Attention, MLA)

> KV 低秩压缩到潜空间，DeepSeek-V2 核心

## 📌 原理与思想

将 KV 先下投影压缩到低维潜空间（存入 Cache），再上投影恢复。压缩比可达 90%+，远超 GQA。

**它解决什么问题**
- GQA 只是让多个 query 头共享同一份 `K/V`，砍的是头的冗余；每个 `K/V` 向量仍然要存满 `Dh` 维，压缩比有下限。
- 长上下文下 KV Cache 仍是显存大头，batch 一大就 OOM —— 头数最多减到 1（MQA），这条路已经走到头了。
- 真正冗余的是「每个 `K/V` 向量本身能由更少的自由度表示」：那就别缓存 `K/V`，改缓存它的低维编码。

**核心思想**
- 缓存的对象从 `K`、`V` 换成一个共享的低维潜向量 `c_kv`，要用的时候再临时上投影恢复。
- 压缩发生在「存储维度」上而不是「头数」上，所以压缩比可以远超 GQA：DeepSeek-V2 的潜维度只有 512。
- `Q` 也做同样的低秩压缩，但只为省训练时的激活显存，它不进 Cache，对推理显存没有贡献。

**算法步骤与推导**
- `c_kv = x·W_dkv` → `[B, S, C]`，`C` 远小于 `H·Dh`；**只有这个进 cache**。
- 用时 `kv = c_kv·W_ukv` → `[B, S, H·(Dh+Dr+Dh)]`，再 split 成 `k_content`、`k_rope`、`v`。
- `k_rope`/`q_rope` 是单独留给 RoPE 的不压缩分量：RoPE 是位置相关的，和内容挤进同一个低秩空间会被压坏。
- `q = cat(q_content, q_rope)`、`k = cat(k_content, k_rope)` 后做 SDPA，之后的流程与普通注意力一致。

**对比与代价**
- 对比 GQA：GQA 的 cache 是 `2·G·Dh`，MLA 是 `C`（DeepSeek-V2 取 512，约等于 4 个 `Dh=128` 头的量），压缩比从 4× 提到 10× 以上。
- 代价是解码时每个 token 都要临时上投影，多了一次矩阵乘；换来的是能开更大的 batch，端到端吞吐反而上升。
- 另一个代价是实现复杂：RoPE 必须拆出来单独走一路，训练时的激活显存还得靠低秩 `Q` 压回去。

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


