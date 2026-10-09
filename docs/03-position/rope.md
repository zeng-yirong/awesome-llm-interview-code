# 旋转位置编码 (Rotary Position Embedding, RoPE)

> 旋转 Q/K 向量注入位置信息，LLaMA 标配

## 📌 原理与思想

### 核心概念
将位置信息编码为旋转角度，对 Q 和 K 的每对相邻维度施加旋转。旋转后 Q·K 的点积自然包含相对位置信息。相比绝对位置编码（无法处理变长序列）和 ALiBi（需要额外参数），RoPE 无需额外参数即可自然编码相对位置。

### 核心思想
通过旋转矩阵将位置信息注入 Q 和 K。对每对相邻维度 [x₁, x₂] 施加旋转 [-x₂, x₁]，等价于旋转 90°。旋转后 q_m · k_n 只依赖 m-n，天然具有相对位置感知能力。只对 Q 和 K 施加旋转，V 不变。

### 算法步骤
1. 预计算频率：inv_freq = 1/(10000^(2i/d))
2. 计算角度：angles = outer(positions, inv_freq)
3. 预计算 cos/sin：cos = cos(angles), sin = sin(angles)
4. 定义 rotate_half：chunk(x, 2, dim=-1) → cat(-x2, x1)
5. 应用旋转：q_rot = q × cos + rotate_half(q) × sin
6. 对 k 同样应用旋转

## 📐 核心公式

```
f(q, m) = q × cos(mθ) + rotate_half(q) × sin(mθ)

rotate_half([x₁, x₂]) = [-x₂, x₁]

等价旋转矩阵: [cos(mθ), -sin(mθ)] [x₁]
              [sin(mθ),  cos(mθ)] [x₂]

θᵢ = 1/10000^(2i/d)
```

## 📊 张量流程图

```
预计算:
  inv_freq = 1/(10000^(2i/d)): [d/2]
  angles = outer(pos, inv_freq): [S, d/2]
  cos, sin = cos(angles), sin(angles): [S, d]
  (复制一份匹配完整维度)

前向:
  q: [B, S, H, D] ──┐
                     ├─ q' = q×cos + rotate_half(q)×sin
  cos,sin: [1,S,1,D] ┘
  k: [B, S, H, D] ──→ k' = k×cos + rotate_half(k)×sin

rotate_half(x):
  x1, x2 = chunk(x, 2, dim=-1)    # 各 [B,S,H,D/2]
  return cat(-x2, x1, dim=-1)     # 旋转90°
```

## 💻 代码实现

```python
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        # 预计算 cos/sin
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, inv_freq)           # [S, D/2]
        angles = torch.cat([angles, angles], dim=-1) # [S, D]
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def forward(self, q, k):
        """q,k: [B, S, H, D]"""
        S = q.size(1)
        cos = self.cos[:S].view(1, S, 1, -1)
        sin = self.sin[:S].view(1, S, 1, -1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

# 测试
if __name__ == "__main__":
    B, S, H, D = 2, 10, 8, 64
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    
    rope = RotaryEmbedding(D)
    q_rot, k_rot = rope(q, k)
    
    print(f"Input q shape: {q.shape}")     # [2, 10, 8, 64]
    print(f"Output q shape: {q_rot.shape}") # [2, 10, 8, 64]
```

## 🎯 面试要点

- **rotate_half**: `chunk` 两半 → `cat(-x2, x1)` = 旋转 90°
- **只对 Q 和 K 施加旋转**，V 不变
- **预计算 cos/sin** 避免重复计算
- **天然具有相对位置感知**: q_m · k_n 只依赖 m-n
- **theta=10000 是标准值**，调整可实现长度外推 (NTK/YaRN)
- **LLaMA, Mistral, Qwen 等主流模型都使用 RoPE**


