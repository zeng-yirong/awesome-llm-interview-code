# 缩放点积注意力 (Scaled Dot-Product Attention)

> `softmax(QKᵀ/√d)V` — 所有注意力的基础

## 📌 原理与思想

计算 Q 和 K 的点积，除以缩放因子 √d_k 后通过 softmax 得到注意力权重，最后加权求和 V。

**为什么需要缩放？**
- 当 d_k 较大时，点积结果也会很大
- 过大的值进入 softmax 后会导致梯度消失
- 除以 √d_k 使方差稳定在 1 附近

这是理解多头注意力（MHA）、分组查询注意力（GQA）、Flash Attention 等所有注意力变体的基础。

## 📐 核心公式

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

其中:
- Q: Query 矩阵 [batch, num_heads, seq_len_q, head_dim]
- K: Key 矩阵 [batch, num_heads, seq_len_k, head_dim]
- V: Value 矩阵 [batch, num_heads, seq_len_k, head_dim]
- d_k: head_dim，缩放因子
```

## 📊 张量流程图

```
Q: [B, H, Sq, D]  ──┐
                    ├─ matmul → [B, H, Sq, Sk] → /√D → mask → softmax → [B, H, Sq, Sk]
K: [B, H, Sk, D]  ──┘                                                          │
                                                                        matmul   │
V: [B, H, Sk, D]  ─────────────────────────────────────────────────────→ [B, H, Sq, D]
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    缩放点积注意力
    
    Args:
        q, k, v: [batch, num_heads, seq_len, head_dim]
        mask: [batch, 1, seq_len_q, seq_len_k] (0=屏蔽)
    
    Returns:
        output: [batch, num_heads, seq_len_q, head_dim]
        attn_weights: [batch, num_heads, seq_len_q, seq_len_k]
    """
    d_k = q.size(-1)
    
    # 步骤1: 计算注意力得分（缩放点积）
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 步骤2: 应用注意力掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 步骤3: Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 步骤4: 加权求和
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights

# 测试
if __name__ == "__main__":
    B, H, S, D = 2, 4, 8, 64
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    
    # 因果掩码（下三角）
    causal_mask = torch.tril(torch.ones(S, S)).view(1, 1, S, S)
    
    output, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    print(f"Output shape: {output.shape}")      # [2, 4, 8, 64]
    print(f"Weights shape: {weights.shape}")    # [2, 4, 8, 8]
```

## 🎯 面试要点

- **缩放因子 1/√d_k**: 当 d_k 大时点积方差大，softmax 进入饱和区梯度消失
- **mask=0 位置填 -1e9**: softmax 后≈0，实现因果/填充屏蔽
- **这是 MHA / GQA / MQA / Flash Attention 的公共基础**
- **时间复杂度**: O(n²d)，空间复杂度 O(n²)
- **面试常问**: 为什么要缩放？不缩放会怎样？

---

**来源**: [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) + [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)
