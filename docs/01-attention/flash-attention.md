# Flash Attention

> 分块计算 + Online Softmax → O(N) 显存

## 📌 原理与思想

将 Q/K/V 分成块，在 SRAM 中完成注意力计算，避免将 O(N²) 的注意力矩阵写入 HBM。利用 Online Softmax 算法，不需要存储完整的注意力矩阵。

**它解决什么问题**
- 标准注意力要把 `[B, H, S, S]` 的分数矩阵写回 HBM 再读回来做 softmax；`S=8192` 时单头就是 6700 万个元素，来回读写三次。
- GPU 的算力远快于显存带宽，这个算子属于典型的 memory-bound：瓶颈全在搬数据，不在乘加。
- SRAM（共享内存）快一个数量级但只有几十 MB，所以问题变成「怎么在不物化完整矩阵的前提下把 softmax 算完」。

**核心思想**
- 把 `Q/K/V` 按块切分，让每个块的分数只在 SRAM 里存在，算完立刻消费掉。
- 难点是 softmax 需要整行的 max 与 sum，而分块后一次只能看到一段。Online Softmax 用「边遍历边修正」解决：每来一个新块就更新 running max，再把之前累积的结果按 `exp(m_old - m_new)` 缩回去。
- 于是每行只需要在寄存器里维护 `m`、`l`、`O` 三个状态，显存占用从 `O(S²)` 降到 `O(S)`。

**算法步骤与推导**
- 把 `Q` 切成 `Tr` 块（常驻 SRAM），`K/V` 切成 `Tc` 块（从 HBM 流式读入）。
- 对每个 `Q` 块遍历所有 `K/V` 块，累加三件事：`m = max(m, rowmax(S_block))`、`l = l·exp(m_old - m) + rowsum(exp(S_block - m))`、`O = O·exp(m_old - m) + exp(S_block - m)·V_block`。
- 三个式子是同一件事：新块到来后先修正旧的归一化因子，再把新块的贡献加进来，所以中途得到的 `O` 始终是「已遍历部分」的正确结果。
- 全部遍历完再统一除以 `l`。结果是精确的 softmax，不是近似 —— 只差浮点累加顺序。

**对比与代价**
- IO 从 `O(S²)` 降到 `O(S²d/M)`：`Q` 块在 SRAM 里被复用，`K/V` 只从 HBM 读一遍。
- 代价是要自己写 CUDA kernel，还要处理掩码、变长等分支；反向也必须重算注意力而不是读回概率矩阵，用额外算力换显存。
- 结果是数值等价的，所以可以逐层替换、不需要重新训练；这也是它能迅速成为标准实现的原因。

## 📐 核心公式

```
标准 Attention:  IO = O(N²)  (存注意力矩阵)
Flash Attention: IO = O(N²d/M)  (M=SRAM大小)

核心: Online Softmax
  m_new = max(m_old, max(S_block))
  l_new = l_old * exp(m_old - m_new) + sum(exp(S_block - m_new))
  O_new = O_old * exp(m_old - m_new) + exp(S_block - m_new) @ V_block
```

## 📊 张量流程图

```
# 标准实现：把中间矩阵物化到 HBM
S = Q·Kᵀ :: [B, H, S, S] :: 一次写完整个分数矩阵
P = softmax(S) :: [B, H, S, S] :: 再整块读回来做 softmax
O = P·V :: [B, H, S, Dh] :: 第三次读写 O(S²) 的数据
$ IO = O(S²)：瓶颈在显存带宽，不在乘加

# Flash Attention：分块 + Online Softmax
+ Q 分块 :: [B, H, Br, Dh] :: 常驻 SRAM，反复使用
+ K, V 分块 :: [B, H, Bc, Dh] :: 从 HBM 流式读入，每个 Q 块读一遍
m = max(m, rowmax(S_block)) :: [B, H, Br] :: 维护每行的 running max
l = l·exp(m_old - m) + rowsum(exp(S_block - m)) :: [B, H, Br] :: 修正后的分母
O = O·exp(m_old - m) + exp(S_block - m)·V_block :: [B, H, Br, Dh] :: 旧结果缩放后叠加新块
> 全程不需要 [S, S] 矩阵，每行只维护 m、l、O 三个状态
$ IO 从 O(S²) 降到 O(S²d/M)，M 是 SRAM 大小；结果与标准 softmax 数值等价
```

## 💻 代码实现

```python
import torch
import math

def flash_attention(Q, K, V, block_size=64):
    """
    Flash Attention 简化实现 (核心思想)
    Q, K, V: [B, H, N, D]
    """
    B, H, N, D = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(B, H, N, 1, device=Q.device)
    m = torch.full((B, H, N, 1), float('-inf'), device=Q.device)

    for i in range(0, N, block_size):
        Qi = Q[:, :, i:i+block_size]
        oi = torch.zeros_like(Qi)
        li = torch.zeros(B, H, Qi.shape[2], 1, device=Q.device)
        mi = torch.full((B, H, Qi.shape[2], 1), float('-inf'), device=Q.device)

        for j in range(0, N, block_size):
            Kj = K[:, :, j:j+block_size]
            Vj = V[:, :, j:j+block_size]
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(D)

            # Online Softmax 更新
            m_new = torch.maximum(mi, Sij.max(dim=-1, keepdim=True).values)
            exp_corr = torch.exp(mi - m_new)
            oi = oi * exp_corr
            li = li * exp_corr
            Pij = torch.exp(Sij - m_new)
            oi = oi + torch.matmul(Pij, Vj)
            li = li + Pij.sum(dim=-1, keepdim=True)
            mi = m_new

        O[:, :, i:i+block_size] = oi / li
    return O

# 测试
if __name__ == "__main__":
    B, H, N, D = 2, 8, 256, 64
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    
    output = flash_attention(Q, K, V, block_size=32)
    print(f"Output shape: {output.shape}")  # [2, 8, 256, 64]
```

## 🎯 面试要点

- **IO-aware**: GPU SRAM 快但小(20MB), HBM 慢但大(40GB)
- **Online Softmax**: 避免存储 N×N 注意力矩阵
- **IO 复杂度从 O(N²) 降到 O(N²d/M)**
- **实际用 CUDA/Triton kernel 实现**，这里展示算法思想
- **Flash Attention 2/3** 进一步优化了并行度和 warp 级别操作
- **面试重点**: 理解分块计算和 Online Softmax 的思想


