# 因果掩码 (Causal Mask)

> 下三角矩阵，防止看到未来信息

## 📌 原理与思想

### 核心概念
在 decoder 中使用下三角矩阵作为 mask，使得位置 i 只能关注位置 ≤i 的 token。这是自回归生成的基础，相比无 mask 的注意力，确保模型在训练时不会"看到未来"。

### 核心思想
通过下三角矩阵屏蔽未来位置的信息，使每个位置只能 attend 到当前及之前的 token。被屏蔽的位置填充 -inf，softmax 后变为 0，从而实现因果约束。

### 算法步骤
1. 创建下三角矩阵：mask = torch.tril(torch.ones(S, S))
2. 调整维度：mask = mask.unsqueeze(0).unsqueeze(0) → [1, 1, S, S]
3. 在注意力计算中应用：scores.masked_fill(mask == 0, -inf)
4. Softmax 归一化：attn = softmax(scores)

## 📐 核心公式

```math
M_{ij}=\begin{cases}1, & j\le i\\[2pt] 0, & j>i\end{cases}
\qquad
S'_{ij}=S_{ij}-\infty\,(1-M_{ij})
```

```math
A=\text{softmax}(S'),\qquad A_{ij}=0\ \text{ for }\ j>i
```

## 📊 张量流程图

```mermaid
flowchart LR
  s["scores = Q·Kᵀ/√D<br/>[B, H, S, S]"] --> mf["masked_fill(mask == 0, -1e9)<br/>[B, H, S, S]"]
  m["mask = tril(ones(S, S))<br/>下三角可见，上三角屏蔽"] --> mf
  mf --> a["A = softmax(·, -1)<br/>exp(-1e9) 下溢为 0"] --> o["O = A·V<br/>[B, H, S, D_h]<br/>第 i 行只混合 j ≤ i"]
```

## 💻 代码实现

```python
import torch

def create_causal_mask(seq_len, device='cpu'):
    """创建因果掩码 (下三角)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

# 在 attention 中使用
def attention_with_causal_mask(q, k, v):
    B, H, S, D = q.shape
    mask = create_causal_mask(S, device=q.device)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

# 测试
if __name__ == "__main__":
    mask = create_causal_mask(4)
    print(mask)
    # tensor([[[[1., 0., 0., 0.],
    #          [1., 1., 0., 0.],
    #          [1., 1., 1., 0.],
    #          [1., 1., 1., 1.]]]])

    # 打印掩码只是看形状；真正要验的是「因果」这条不变量：
    # 改掉位置 t 之后的 token，位置 ≤ t 的输出必须一个字节都不变
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 5, 8
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    out = attention_with_causal_mask(q, k, v)
    v_future = v.clone()
    v_future[:, :, 3:] += 100.0          # 只动位置 3 及之后的 V
    out_future = attention_with_causal_mask(q, k, v_future)
    print("改未来 V 后前 3 个位置的输出变化 (应为 0):",
          (out[:, :, :3] - out_future[:, :, :3]).abs().max().item())

    # 对照: 不加掩码，未来的 V 会一路影响前面所有位置
    def attn_no_mask(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    print("无掩码时同样的变化量 (非 0):",
          (attn_no_mask(q, k, v)[:, :, :3]
           - attn_no_mask(q, k, v_future)[:, :, :3]).abs().max().item())
    # 之所以成立，是因为 A_ij 在 j > i 处置成了 0：第 0 行的注意力全给了第 0 个 token
```

## 🎯 面试要点

- **torch.tril**: 生成下三角矩阵
- **被屏蔽位置填 -inf**: softmax 后为 0
- **decoder-only (GPT)**: 全程使用 causal mask
- **训练时可用 causal mask 并行计算所有位置的 loss**
- **面试常问**: 为什么训练时可以并行？因为 causal mask 保证了每个位置只能看到之前的 token
- **推理时**: 每步只有 1 个 token，掩码自动失效 —— 训练与推理走的是同一套代码

