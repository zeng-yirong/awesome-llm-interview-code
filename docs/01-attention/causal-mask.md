# 因果掩码 (Causal Mask)

> 下三角矩阵，防止看到未来信息

## 📌 原理与思想

在 decoder 中使用下三角矩阵作为 mask，使得位置 i 只能关注位置 ≤i 的 token。这是自回归生成的基础。

**它解决什么问题**
- 自回归的训练目标是「预测第 i 个 token 时只能看到前 i 个」；但注意力天生是全局的，位置 0 会直接看到位置 9 的答案，等于把标签喂给了模型。
- 如果老老实实按因果顺序一次前向一个位置，训练要跑 `S` 遍前向，完全无法并行，等于放弃 GPU。
- 掩码让「一次前向算完所有位置」与「每个位置只看到自己的前缀」同时成立 —— 这是 GPT 类模型能高效训练的根基。

**核心思想**
- 在 softmax 之前把 `j > i` 的位置填成一个极大的负数，softmax 之后这些位置的权重正好下溢为 0。
- 下三角矩阵只是「允许看谁」的形状描述，实现上就是对 score 矩阵做一次 `masked_fill`。
- 输出形状与普通注意力完全一致，因果性只由 mask 的取值保证，不改变任何一步的形状。

**算法步骤与推导**
- 构造 `mask = tril(ones(S, S))`：1 表示可见（下三角含对角线），0 表示屏蔽（上三角）。
- `scores = QKᵀ/√D` → `[B, H, S, S]`，把 `mask == 0` 的位置填成 `-1e9`。
- 沿最后一维 softmax：被填的位 `exp(-1e9)` 下溢为 0，权重精确为 0，其余位置按剩余项重新归一化。
- `O = A·V`，第 `i` 行只混合了 `j ≤ i` 的 `V` —— 因果性由构造保证，不需要额外检查。

**对比与代价**
- 与 padding mask 的区别：padding mask 屏蔽「不存在的 token」，因果 mask 屏蔽「未来的 token」，两者通常叠加成同一个二维掩码。
- 填 `-1e9` 而不是 `-inf`：整行都被屏蔽时 `softmax(-inf)` 会得到 NaN，`-1e9` 在 fp32 下已经下溢到 0，行为更稳。
- 代价是 softmax 每行的有效长度不同，无法完全均匀分块；Flash Attention 要专门判断「当前块是否跨越对角线」，跨了就多做一次掩码。

## 📐 核心公式

```
mask[i][j] = 1  if j ≤ i
           = 0  if j > i

scores = scores.masked_fill(mask == 0, -inf)
attn = softmax(scores, dim=-1)
```

## 📊 张量流程图

```
# 只改 softmax 的输入，不改任何形状
scores = Q·Kᵀ/√D :: [B, H, S, S] :: 与普通注意力完全一样
mask = tril(ones(S, S)) :: [S, S] :: 下三角含对角线为 1（可见），上三角为 0（屏蔽）
scores.masked_fill(mask == 0, -1e9) :: [B, H, S, S] :: 屏蔽位填一个极大的负数
A = softmax(scores, -1) :: [B, H, S, S] :: exp(-1e9) 下溢为 0，权重精确为 0
O = A·V :: [B, H, S, Dh] :: 第 i 行只混合了 j ≤ i 的 V
$ 形状全程不变，因果性完全由 mask 的取值保证
> 推理时每步只有 1 个 token，掩码自动失效 —— 训练与推理走的是同一套代码
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


