# 标准前馈网络 (Feed-Forward Network)

> 两层 MLP，占 Transformer 2/3 参数量

## 📌 原理与思想

注意力层之后的两层全连接网络。先上投影扩展维度（通常 4 倍），应用激活函数，再下投影恢复维度。

**它解决什么问题**
- 注意力本质是加权求和，对 `V` 是线性的；没有 FFN，整个 Transformer 就退化成多层线性变换的叠加，表达力约等于单层。
- 注意力负责 token 之间的信息交换，不做单个 token 内部的非线性加工 —— 这件事必须由别的模块来做。
- FFN 也是模型存知识的地方：`D=4096` 时它占 `8D²` 参数，是注意力 `4D²` 的两倍，约占整个 Transformer 的 2/3。

**核心思想**
- 先用一个大矩阵把 `D` 维升到 `4D`，在更高维的空间里做非线性，再压回 `D` 维。
- 升维—非线性—降维，等于在中间层获得一个更宽的特征加工区；逐 token 独立，同一个 `W₁` 作用在所有位置上。

**算法步骤与推导**
- 上投影 `W₁: [D, 4D]` 得到 `[B, S, 4D]`，这一步只有乘加，不含非线性。
- 过 `ReLU` 逐元素取 `max(0, x)`：这是整个模块里唯一引入非线性的地方。
- 下投影 `W₂: [4D, D]` 回到 `[B, S, D]`，与残差流维度对齐，才能和输入相加。
- 形状全程是 `[B, S, ·]`，只有最后一维在 `D` 与 `4D` 之间来回，序列维和 batch 维完全不动。

**对比与代价**
- 对比 SwiGLU：门控版把参数拆成三个矩阵、效果更好，代价是多一个矩阵，所以中间维度从 `4D` 缩到 `8/3·D` 保持总参数量持平。
- 代价是参数量和计算量都集中在这里：4 倍扩展意味着 `8D²` 参数，推理时这部分是主要的 FLOPs 来源。
- `ReLU` 在负区间梯度恒为 0，神经元一旦长期落在负区间就再也学不动（dead neuron）；`GELU`、`SiLU` 更平滑，现在的模型基本都换掉了 `ReLU`。

## 📐 核心公式

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂

参数量: D × 4D + 4D × D = 8D² (vs Attention: 4D²)

其中:
- W₁: [D, 4D] 上投影
- W₂: [4D, D] 下投影
- 中间维度通常是 d_model 的 4 倍
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


