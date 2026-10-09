# 多头注意力 (Multi-Head Attention)

> 并行多头 → 拼接 → 输出投影

## 📌 原理与思想

将输入投影到多个子空间，每个头独立计算注意力，最后拼接并通过线性层融合。不同头可学习不同的注意力模式。

**它解决什么问题**
- 单头注意力只有一组 `Q/K/V` 投影，softmax 权重会把「关注哪儿」这件事压成一种模式 —— 同一个位置没法同时对几个不同的位置组合分配注意力。
- 所有维度还共用同一套投影权重，无法让不同维度去捕捉不同类型的关系（句法、指代、局部顺序）。
- 一个位置常常需要同时盯住多个对象，这是单头结构上做不到的事。

**核心思想**
- 把 `D` 维拆成 `H` 个 `Dh = D/H` 维的子空间，每个子空间各自独立做一次完整的注意力。
- 每个头有自己的 `W_q/W_k/W_v`，等于把「关注什么」也交给模型自己学。
- 最后 `concat` 回 `D` 维再过 `W_o` 融合；`W_o` 是唯一发生跨头交互的地方。

**算法步骤与推导**
- `x: [B, S, D]` 分乘三个投影得到 `[B, S, D]`，再 view 成 `[B, S, H, Dh]`、transpose 成 `[B, H, S, Dh]`。
- transpose 只是把头维提到前面，让每个头在最后两维上是一个独立的矩阵，从而能一次性批量做 SDPA。
- 每个头独立算 `softmax(QKᵀ/√Dh)·V`，归一化只在 `Dh` 内、每个头各自进行，头与头之间不共享分母。
- transpose + view 变回 `[B, S, D]`，过 `W_o` 得到输出。总计算量仍是 `O(S²·D)`：`H` 个头各算 `S²·Dh`，加起来正好等于 `S²·D`。

**对比与代价**
- 参数量不随 `H` 增长（`H·Dh = D`），多头只多出一组中间张量，代价几乎为零。
- 代价在显存：中间注意力矩阵是 `[B, H, S, S]`，与 `H` 成正比，KV Cache 同理 —— GQA、MQA 砍的正是这一项。
- 头也不是越多越好：`Dh` 太小时单个头的表达空间受限，实践中 `Dh` 一般取 64~128。

## 📐 核心公式

$$
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}\!\left(\text{head}_1,\dots,\text{head}_H\right)W_O
$$

$$
\text{head}_i=\operatorname{Attention}\!\left(QW_i^{Q},\ KW_i^{K},\ VW_i^{V}\right),
\qquad
D_h=\frac{D}{H}
$$

## 📊 张量流程图

```
# 一次投影，再把 D 拆成 H 个头
x :: [B, S, D] :: 输入
+ Q = x·W_qᵀ :: [B, S, D] :: 查询投影
+ K = x·W_kᵀ :: [B, S, D] :: 键投影
+ V = x·W_vᵀ :: [B, S, D] :: 值投影
view + transpose :: [B, H, S, Dh] :: 把 D 拆成 H×Dh，再把头维提到前面
SDPA :: [B, H, S, Dh] :: 每个头独立算一次缩放点积注意力
transpose + view :: [B, S, D] :: 头拼回完整的 D 维
y = ·W_o :: [B, S, D] :: 输出投影，唯一发生跨头交互的地方
$ H·Dh = D，参数量与单头完全相同，只多出一组中间张量
> 中间的注意力矩阵是 [B, H, S, S]，显存与头数成正比 —— GQA、MQA 砍的就是这一项
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


