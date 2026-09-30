# Flash Attention

> 分块计算 + Online Softmax → O(N) 显存

## 📌 原理与思想

将 Q/K/V 分成块，在 SRAM 中完成注意力计算，避免将 O(N²) 的注意力矩阵写入 HBM。利用 Online Softmax 算法，不需要存储完整的注意力矩阵。

**为什么需要 Flash Attention？**
- 标准 Attention: IO = O(N²)（存注意力矩阵）
- Flash Attention: IO = O(N²d/M)（M=SRAM大小）
- GPU SRAM 快但小(20MB), HBM 慢但大(40GB)

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
标准 Attention:                Flash Attention:
  Q → scores[N×N] → softmax → @V    Q分块 → 逐块计算 → Online Softmax → 输出
      ↑ 写入HBM (O(N²))                    ↑ 只在SRAM (O(N))

Online Softmax 三步:
  Pass 1: 找每行最大值 m (数值稳定)
  Pass 2: 计算 exp(x-m) 的和 l (分母)
  Pass 3: 计算 exp(x-m)/l * V (分子)
  → 不需要存储完整 N×N 矩阵!
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

---

**来源**: [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)
