# 分组查询注意力 (Grouped Query Attention, GQA)

> 多 Q 头共享 KV 头，LLaMA 2 标配

## 📌 原理与思想

### 核心概念
MHA 和 MQA 的折中方案：Q 有 H 个头，KV 只有 G 个头 (G<H)。多个 Q 头共享同一组 KV 头，大幅减少 KV Cache。相比 MHA 节省推理显存，相比 MQA 保持更好的模型质量。

### 核心思想
通过让多个 Q 头共享同一组 KV 头，在保持模型表达能力的同时大幅减少 KV Cache 大小。核心操作是 repeat_kv：将 G 个 KV 头复制扩展为 H 个，以匹配 Q 的头数。

### 算法步骤
1. 线性投影：Q → [B, S, H, Dh]，K,V → [B, S, G, Dh]
2. 复制 KV 头：repeat_kv(K, H/G) → [B, S, H, Dh]
3. 转置维度：transpose(1, 2) → [B, H, S, Dh]
4. 计算缩放点积注意力
5. 合并多头并输出投影

## 📐 核心公式

```
Q: [B, H, S, Dh]     (H 个头)
K,V: [B, G, S, Dh]   (G 个头, G < H)

repeat_kv: K,V → [B, H, S, Dh]   (复制 G→H)

对比:
- MHA: G = H (每个 Q 头独立 KV)
- MQA: G = 1 (所有 Q 头共享一组 KV)
- GQA: 1 < G < H (折中方案)
```

## 📊 张量流程图

```
Q: [B, H, S, Dh]  ─────────────────────────┐
                                            │ SDPA
K: [B, G, S, Dh] → repeat_kv → [B, H, S, Dh] ─┤
V: [B, G, S, Dh] → repeat_kv → [B, H, S, Dh] ─┘

repeat_kv 实现:
  x: [B, G, S, Dh]
  → x[:,:,None,:,:]          [B, G, 1, S, Dh]
  → .expand(B, G, H/G, S, Dh) [B, G, H/G, S, Dh]
  → .reshape(B, H, S, Dh)    [B, H, S, Dh]
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.g = n_heads, n_kv_heads
        self.dh = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model)

    def repeat_kv(self, x, n_rep):
        """[B, G, S, Dh] → [B, H, S, Dh]"""
        if n_rep == 1:
            return x
        B, G, S, D = x.shape
        return (x[:, :, None, :, :]
                 .expand(B, G, n_rep, S, D)
                 .reshape(B, G * n_rep, S, D))

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, S, self.g, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, S, self.g, self.dh).transpose(1, 2)
        
        # 核心：复制 KV 以匹配 Q 的头数
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    gqa = GQA(D, n_heads=32, n_kv_heads=8)  # 4 个 Q 头共享 1 个 KV 头
    out = gqa(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 512]
    print(f"Output shape: {out.shape}") # [2, 10, 512]
```

## 🎯 面试要点

- **repeat_kv**: `unsqueeze → expand → reshape`，面试必写
- **KV Cache 减少到 G/H × 100%**（如 8/32 = 25%）
- **MHA: G=H; MQA: G=1; GQA: 1<G<H**
- **LLaMA 2 70B 用 GQA(8 groups)**，推理内存降 75%
- **训练速度几乎不受影响，推理速度显著提升**


