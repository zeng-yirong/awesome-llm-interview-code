# 因果掩码 (Causal Mask)

> 下三角矩阵，防止看到未来信息

## 📌 原理与思想

在 decoder 中使用下三角矩阵作为 mask，使得位置 i 只能关注位置 ≤i 的 token。这是自回归生成的基础。

**为什么需要因果掩码？**
- 自回归模型在生成时只能看到之前的 token
- 训练时通过因果掩码模拟这一过程
- 使得模型可以并行训练所有位置

## 📐 核心公式

```
mask[i][j] = 1  if j ≤ i
           = 0  if j > i

scores = scores.masked_fill(mask == 0, -inf)
attn = softmax(scores, dim=-1)
```

## 📊 张量流程图

```
seq_len = 4 的 causal mask (1=可见, 0=屏蔽):

  ┌           ┐
  │ 1  0  0  0 │   row 0 只能看自己
  │ 1  1  0  0 │   row 1 看 0,1
  │ 1  1  1  0 │   row 2 看 0,1,2
  │ 1  1  1  1 │   row 3 看所有
  └           ┘
  = torch.tril(torch.ones(S, S))
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
```

## 🎯 面试要点

- **torch.tril**: 生成下三角矩阵
- **被屏蔽位置填 -inf**: softmax 后为 0
- **decoder-only (GPT)**: 全程使用 causal mask
- **训练时可用 causal mask 并行计算所有位置的 loss**
- **面试常问**: 为什么训练时可以并行？因为 causal mask 保证了每个位置只能看到之前的 token

---

**来源**: [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)
