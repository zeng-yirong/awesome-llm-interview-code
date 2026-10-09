# 旋转位置编码 (Rotary Position Embedding, RoPE)

> 旋转 Q/K 向量注入位置信息，LLaMA 标配

## 📌 原理与思想

将位置信息编码为旋转角度，对 Q 和 K 的每对相邻维度施加旋转。旋转后 Q·K 的点积自然包含相对位置信息。

**它解决什么问题**
- 注意力对位置完全无知：把输入序列打乱，`Q·Kᵀ` 的结果一模一样 —— 没有位置编码，模型连「谁在谁前面」都不知道。
- 可学习的绝对位置编码受限于训练时见过的最大长度，超出就 OOD；而且位置与内容被焊在同一组参数上，外推只能靠插值再微调。
- 理想的位置信息应该只以「相对距离」的形式进入点积，这样训练时没见过的长度也能自然泛化。

**核心思想**
- 给 `q`、`k` 各乘一个随位置变化的旋转矩阵 `R_m`，那么 `(R_m q)·(R_n k) = qᵀ·R_{n-m}·k` —— 点积里只剩相对位置，绝对位置被自动消掉。
- 旋转矩阵是正交的，不改变向量长度，原有的数值尺度不受影响。
- 把 `D` 维拆成 `D/2` 个二维平面，每个平面用不同频率 `θᵢ` 旋转：高频维度分辨近邻，低频维度分辨远距离。

**算法步骤与推导**
- 预计算 `inv_freq = 1/10000^(2i/d)`（`i = 0..d/2-1`），与位置做外积得到角度矩阵 `[S, d/2]`，取 cos/sin 后复制拼接成 `[S, d]`。
- 拼成 `[S, d]` 是为了直接和 `[B, S, H, D]` 广播，不用在 head 维上重复展开。
- `rotate_half(x)` 把后半段取负拼到前面，配合 cos/sin 正好等价于乘一个二维旋转矩阵，这样不必真的构造 `D×D` 的稀疏旋转矩阵。
- 只作用在 `q` 和 `k` 上，不动 `v` —— `v` 是被加权求和的内容，本身与位置无关。

**对比与代价**
- 对比可学习绝对位置编码：零新增参数、天然相对、还能靠改 `inv_freq` 做外推（NTK-aware、线性插值）。
- 对比 ALiBi：ALiBi 直接在注意力分数上加线性偏置，实现更简单；RoPE 把位置编进 `Q/K` 的方向里，与内容耦合更紧，是目前的主流选择。
- 代价是旋转改变了向量方向，后续线性层得重新适应；而且基频 `10000` 是超参，直接外推到远超训练长度时高频维度会震荡，才需要专门的插值技巧。

## 📐 核心公式

$$
f(q,m)=q\odot\cos(m\theta)+\operatorname{rotate\_half}(q)\odot\sin(m\theta)
$$

$$
\operatorname{rotate\_half}([x_1,x_2])=[-x_2,\ x_1],
\qquad
R(m\theta)=\begin{pmatrix}\cos m\theta & -\sin m\theta\\[2pt] \sin m\theta & \cos m\theta\end{pmatrix}
$$

$$
(R_mq)\cdot(R_nk)=q^{\top}R_{n-m}\,k
$$

$$
\theta_i=10000^{-2i/d},\qquad i=0,\dots,\frac{d}{2}-1
$$

## 📊 张量流程图

```
# 预计算：每个位置、每个维度对对应的旋转角
inv_freq = 1/10000^(2i/d) :: [d/2] :: i = 0..d/2-1，靠后的维度对频率更低
angles = outer(pos, inv_freq) :: [S, d/2] :: 位置 × 维度对
cos, sin = cos(angles), sin(angles) :: [S, d] :: 复制一份拼成完整维度，好与 [B,S,H,D] 广播

# 旋转：只作用在 Q、K 上
rotate_half(x) :: cat(-x₂, x₁) :: 后半段取负拼到前面，等价于旋转 90°
+ q' = q·cos + rotate_half(q)·sin :: [B, S, H, D] :: 查询被旋转
+ k' = k·cos + rotate_half(k)·sin :: [B, S, H, D] :: 键被旋转同样的角度
> v 不参与旋转：它是被加权求和的内容，本身与位置无关
$ (R_m q)·(R_n k) = qᵀ·R_{n-m}·k —— 点积里只剩相对位置 n - m
> 基频 10000 是超参，直接外推到远超训练长度时高频维度会震荡，所以有 NTK-aware、线性插值等改法
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


