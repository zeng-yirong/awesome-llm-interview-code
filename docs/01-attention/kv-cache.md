# KV Cache

> 缓存历史 KV，避免自回归重复计算

## 📌 原理与思想

自回归生成时，每步只处理新 token，但需要与所有历史 token 做注意力。KV Cache 缓存历史的 K/V，避免重复计算。

**它解决什么问题**
- 自回归解码每步只新增 1 个 token，可这个新 token 要和全部历史做注意力；不缓存的话，第 `t` 步就得重算前 `t` 个 token 的 `K/V`。
- 总代价从 `O(S)` 变成 `Σt = O(S²)` 次投影计算，生成 4096 个 token 时白算了约两千倍。
- `K/V` 只依赖 token 自身和它的位置，不随「后面来了什么」改变 —— 这正是它可以被缓存的前提（`Q` 不具备这个性质）。

**核心思想**
- 用一个显式张量保存每层历史的 `K/V`，新 token 只算自己的 `K/V`，再拼到末尾。
- `Q` 不缓存：每一步只有最新的 `Q` 有用，历史 `Q` 不会再被任何计算用到。
- 空间换时间：显存随序列长度线性增长，换来每步计算量恒定。

**算法步骤与推导**
- Prefill：整段 prompt 一次前向，把每层的 `K/V` 写进 cache，形状 `[B, H_kv, S_prompt, Dh]`。
- Decode：新 token 算出 `q/k/v`，形状都是 `[B, H, 1, ·]`；只把 `k`、`v` 追加到 cache 末尾。
- 用 `[B, H, 1, Dh]` 的 `q` 去和 `[B, H, S+t, Dh]` 的 cache 做 SDPA，一步得到一个输出 token，计算量恒定。
- cache 是逐 token 增长的，朴素实现要么预分配最大长度（浪费），要么整块复制（拷贝开销）—— 这正是 PagedAttention 要解决的问题。

**对比与代价**
- 缓存大小 = `2 × n_layers × n_kv_heads × S × Dh × 字节数`；LLaMA 2 70B 在 4096 长度、fp16 下约 80GB，比模型本身还大。
- 代价是显存成为首要瓶颈：batch size 和上下文长度都被它卡住，所以才有了 MQA/GQA（减头数）和 MLA（压缩存储）。
- 每步都要读写整个 cache，decode 阶段是 memory-bound、算力大量闲置；把多个请求 batch 起来正是为了填满这个空。

## 📐 核心公式

$$
\text{KV cache}=2\,n_{\text{layers}}\,n_{\text{kv}}\,S\,D_h\,b
$$

$$
\text{LLaMA 2 70B:}\quad 2\times 80\times 8\times 4096\times 128\times 2\ \mathrm{B}=80\ \mathrm{GB}
$$

$$
\text{decode}: \quad K\leftarrow\operatorname{concat}(K,\,k_{\text{new}}),\qquad
V\leftarrow\operatorname{concat}(V,\,v_{\text{new}})
$$

$$
\text{no cache}: \sum_{t=1}^{S}t=O(S^{2}),\qquad
\text{cache}: O(S)\ \text{per step}
$$

## 📊 张量流程图

```
# Prefill：整段算完，把每层的 K/V 留下来
prompt :: [1, S_prompt, D] :: 整段输入，一次前向
Q, K, V 投影 :: [1, H_kv, S_prompt, Dh] :: 只有 K、V 会被留下
cache :: [L, 2, B, H_kv, S, Dh] :: 每层一份，显存随序列长度线性增长

# Decode：每步只算新 token，历史 KV 从缓存拼
token_t :: [1, 1, D] :: 当前步唯一的新输入
+ q_t :: [1, H, 1, Dh] :: 查询，只有这一步用得上，不缓存
+ k_t, v_t :: [1, H_kv, 1, Dh] :: 键值，追加到缓存末尾
K = cat(K_cache, k_t) :: [1, H_kv, S+t, Dh] :: 拼接即可，历史部分完全不重算
O = SDPA(q_t, K, V) :: [1, H, S+t, Dh] :: 一个 query 对整个历史
$ 每步计算量恒定，不缓存则是 Σt = O(S²)
> LLaMA 2 70B 在 4096 长度、fp16 下缓存约 80GB —— 比模型本身还大
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCacheAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        """
        x: [batch, seq_len, d_model]
        kv_cache: (cached_k, cached_v) 各为 [B, H, past_len, Dh]
        """
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, S, self.h, self.dh).transpose(1, 2)

        # 核心：拼接历史 KV
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out), new_cache

# 测试
if __name__ == "__main__":
    B, S, D, H = 1, 10, 512, 8
    attn = KVCacheAttention(D, H)
    
    # Prefill: 处理整个 prompt
    x = torch.randn(B, S, D)
    out, cache = attn(x)
    print(f"Prefill output: {out.shape}")  # [1, 10, 512]
    print(f"Cache K shape: {cache[0].shape}")  # [1, 8, 10, 64]
    
    # Decode: 逐个生成 token
    new_token = torch.randn(B, 1, D)
    out, cache = attn(new_token, kv_cache=cache)
    print(f"Decode output: {out.shape}")  # [1, 1, 512]
    print(f"Cache K shape: {cache[0].shape}")  # [1, 8, 11, 64]
```

## 🎯 面试要点

- **空间换时间**: 缓存 O(L × H × Dh) 显存
- **Prefill 处理整个 prompt，Decode 逐个生成**
- **GQA 减少 KV Cache** (多个 Q 共享 KV)
- **长序列推理的主要显存瓶颈**
- **PagedAttention (vLLM)** 用分页管理 KV Cache 减少碎片
- **面试常问**: KV Cache 有多大？如何优化？


