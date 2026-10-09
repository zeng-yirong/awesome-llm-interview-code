# 标准前馈网络 (Feed-Forward Network)

> 两层 MLP，占 Transformer 2/3 参数量

## 📌 原理与思想

### 核心概念
注意力层之后的两层全连接网络。先上投影扩展维度（通常 4 倍），应用激活函数，再下投影恢复维度。Attention 捕捉 token 间的关系，FFN 对每个 token 独立做非线性变换，占 Transformer 参数量的 2/3。

### 核心思想
通过上投影将维度从 D 扩展到 4D，在高维空间做非线性变换，再下投影回 D 维。这种"扩展-变换-压缩"的结构增强了模型的表达能力。

### 算法步骤
1. 上投影：h = W₁ · x + b₁，维度 D → 4D
2. 激活函数：a = ReLU(h)
3. 下投影：output = W₂ · a + b₂，维度 4D → D

## 📐 核心公式

```math
\text{FFN}(x)=W_2\text{ReLU}(W_1x+b_1)+b_2
```

```math
W_1\in\mathbb{R}^{D\times 4D},\qquad W_2\in\mathbb{R}^{4D\times D}
```

```math
\begin{aligned}
\text{FFN}: &\quad D\cdot 4D+4D\cdot D=8D^{2}\\[2pt]
\text{Attention}: &\quad 4D^{2}
\end{aligned}
```

## 📊 张量流程图

```
# 两层全连接 + 中间一层逐元素非线性
x :: [B, S, D] :: 残差流的输入
h = x·W₁ᵀ :: [B, S, 4D] :: 上投影，中间维度通常取 4 倍
a = ReLU(h) :: [B, S, 4D] :: 逐元素 max(0, x)，唯一引入非线性的地方
y = a·W₂ᵀ :: [B, S, D] :: 下投影回残差流维度，才能与输入相加
$ 参数量 8D²，是注意力 4D² 的两倍，约占 Transformer 的 2/3
> 逐 token 独立：同一个 W₁ 作用在所有位置上，没有任何跨 token 交互
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """x: [B, S, D]"""
        return self.w2(F.relu(self.w1(x)))

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    ffn = FFN(D)
    out = ffn(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 512]
    print(f"Output shape: {out.shape}") # [2, 10, 512]
    
    # 统计参数量
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"FFN params: {total_params:,}")  # 约 4M
```

## 🎯 面试要点

- **中间维度通常 4× d_model**
- **占 Transformer 参数量的 2/3**
- **现代 LLM 多用 SwiGLU 替代标准 FFN+ReLU**
- **MoE 本质上是对 FFN 层的稀疏化**
- **面试常问**: FFN 的作用是什么？为什么参数量这么大？


