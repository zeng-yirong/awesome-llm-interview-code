# 分组查询注意力 (Grouped Query Attention, GQA)

> 多 Q 头共享 KV 头，LLaMA 2 标配

## 📌 原理与思想

MHA 和 MQA 的折中方案：Q 有 H 个头，KV 只有 G 个头 (G<H)。多个 Q 头共享同一组 KV 头，大幅减少 KV Cache。

**为什么需要 GQA？**
- MHA: KV Cache 大，推理慢
- MQA: KV Cache 小，但模型质量下降
- GQA: 在两者之间取得平衡

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

---

**来源**: [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) + [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)
