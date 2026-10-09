# KV Cache

> 缓存历史 KV，避免自回归重复计算

## 📌 原理与思想

### 核心概念
自回归生成时，每步只处理新 token，但需要与所有历史 token 做注意力。KV Cache 缓存历史的 K/V，避免重复计算。相比每步重新计算所有 token 的 O(N²) 复杂度，KV Cache 将其降至 O(N)，是推理加速的核心技术。

### 核心思想
空间换时间：缓存历史 token 的 K/V，每步只计算新 token 的 K/V，然后与缓存拼接。代价是额外的显存占用 O(L × H × Dh)。

### 算法步骤
1. **Prefill 阶段**：处理整个 prompt，计算并缓存所有 KV
2. **Decode 阶段**：每步只处理 1 个新 token
3. 计算新 token 的 Q, K, V
4. 拼接历史 KV：K = cat(K_cache, K_new)
5. 计算注意力：attn(Q, K, V)
6. 更新 cache，输出下一个 token

## 📐 核心公式

```
Prefill:  处理整个 prompt → 缓存所有 KV
Decode:   每步只处理 1 token → K_new = cat(K_cache, K_new)

KV Cache 大小 = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes

例如 LLaMA 2 70B:
  80 layers × 8 kv_heads × 4096 seq_len × 128 head_dim × 2 bytes (fp16)
  = 80 GB!
```

## 📊 张量流程图

```
Prefill 阶段 (处理 prompt):
  prompt [1, S_prompt, D] → 模型 → 缓存 KV [1, H, S_prompt, Dh]

Decode 阶段 (逐 token 生成):
  token_t [1, 1, D] → Q,K,V
  K = cat(K_cache, K_t)    → [1, H, S_prompt+t, Dh]
  V = cat(V_cache, V_t)    → [1, H, S_prompt+t, Dh]
  attn(Q, K, V) → 输出 → 更新 cache
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


