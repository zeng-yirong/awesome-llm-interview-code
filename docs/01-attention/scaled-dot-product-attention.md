# 缩放点积注意力 (Scaled Dot-Product Attention)

> `softmax(QKᵀ/√d)V` — 所有注意力的基础

## 📌 原理与思想

计算 Q 和 K 的点积，除以缩放因子 √d_k 后通过 softmax 得到注意力权重，最后加权求和 V。缩放因子防止点积过大导致 softmax 梯度消失。

**它解决什么问题**
- 全连接层做不到「按内容检索」，参数量还随序列长度增长；点积注意力用一次矩阵乘法算出所有位置对的相关性，零参数、完全可并行。
- 但不缩放会出事：q、k 各维方差为 1 时点积方差恰好等于 `d_k`，`d_k=512` 时标准差约 23。softmax 输入跨度过大会饱和成 one-hot，梯度趋近 0。

**核心思想**
- q 与 k 的点积衡量相关性，softmax 把相关性归一化成权重，再对 v 加权求和。
- 除以 `√d_k` 把点积方差拉回 1，让 softmax 落在梯度健康的区间。

**算法步骤与推导**
- S = QKᵀ → 除 `√d_k` → 加掩码 → softmax 得权重 A → O = AV，形状全程是 `[B, H, Sq, Sk]` 这一族。
- 掩码填 `-1e9` 而不是 0，是为了 softmax 之后权重正好变成 0。

**对比与代价**
- 同分类后续题目都在改它：MHA 拆头、GQA/MQA 复用 KV、Flash Attention 改 IO、MLA 改 KV 存储。
- 代价是 S 必须显式物化成 `[B, H, Sq, Sk]`，显存 O(S²)；而且 softmax 要整行归约，朴素实现无法流式处理。

## 📐 核心公式

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

$$
\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad
\operatorname{Var}(q\cdot k)=d_k,\qquad
Q,K,V\in\mathbb{R}^{n\times d_k},\qquad
d_k=\frac{D}{H}
$$

## 📊 张量流程图

```
# 打分：一次 matmul 得到所有位置对的相关性
+ Q :: [B, H, Sq, D] :: 查询
+ K :: [B, H, Sk, D] :: 键
S = Q·Kᵀ/√D :: [B, H, Sq, Sk] :: 除以 √D 把点积方差从 D 拉回 1
Mask :: 因果 / padding 位置填 -1e9，softmax 后权重≈0
A = softmax(S) :: [B, H, Sq, Sk] :: 每行和为 1

# 加权求和
+ A :: [B, H, Sq, Sk] :: 注意力权重
+ V :: [B, H, Sk, D] :: 值
O = A·V :: [B, H, Sq, D] :: 与 Q 同形
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


