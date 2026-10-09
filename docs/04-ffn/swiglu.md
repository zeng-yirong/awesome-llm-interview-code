# SwiGLU

> 门控 + SiLU 激活，LLaMA/PaLM 标配

## 📌 原理与思想

引入门控机制：一个分支用 SiLU 激活作为门，另一个分支无激活，两者逐元素相乘后下投影。比标准 FFN 效果更好。

**它解决什么问题**
- `ReLU` 只做「截断」，没有任何机制决定「哪些信息该通过」；每一维的增益是固定的，不随输入变化。
- 乘法门控带来的是二阶交互：门与内容相乘，相当于让网络自己学会一个「由输入决定」的缩放系数。
- 实验一致显示，同等参数量下门控 FFN 的 loss 更低，这个收益稳定且可复现。

**核心思想**
- 用两路上投影，一路过 `SiLU` 当门，一路保持线性，两者逐元素相乘再下投影。
- `SiLU(x) = x·σ(x)` 自带门控：`σ(x)` 是 0~1 的开关，`x` 是内容，负区间先降后回升，不会像 `ReLU` 那样被一刀切死。

**算法步骤与推导**
- `gate = SiLU(x·W_gateᵀ)` → `[B, S, d_ff]`，`up = x·W_upᵀ` → `[B, S, d_ff]`，两路必须同维，乘法才是逐元素对齐的。
- `g = gate ⊙ up` 逐元素相乘：门决定每一维通过多少，内容原样保留。
- `y = g·W_downᵀ` → `[B, S, D]`。`d_ff` 取 `8/3·D`，三个矩阵合起来 `3·D·d_ff ≈ 8D²`，正好和标准 FFN 的参数量持平。

**对比与代价**
- 对比标准 FFN：多一个矩阵换到更低的 loss；为了参数量对齐，中间维度从 `4D` 缩到 `8/3·D`，实际宽度反而变小了。
- 代价是矩阵乘法从 2 个变 3 个，算子数量增加，推理时略慢一点点，但相比收益可以忽略。
- LLaMA、PaLM 都用它替换 `ReLU` FFN，`GELU` 版的 GeGLU 同理。

## 📐 核心公式

$$
\operatorname{SwiGLU}(x)=W_{\text{down}}\!\left(\operatorname{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\right)
$$

$$
\operatorname{SiLU}(x)=x\,\sigma(x),\qquad d_{ff}=\tfrac{8}{3}D
$$

$$
\begin{aligned}
\text{SwiGLU}: &\quad 3D\,d_{ff}\approx 8D^{2}\\[2pt]
\text{FFN}: &\quad 8D^{2}
\end{aligned}
$$

## 📊 张量流程图

```
# 门控 FFN：两路上投影，一路当门
x :: [B, S, D] :: 残差流的输入
+ gate = SiLU(x·W_gateᵀ) :: [B, S, d_ff] :: 门，过激活
+ up = x·W_upᵀ :: [B, S, d_ff] :: 内容，不过激活
g = gate ⊙ up :: [B, S, d_ff] :: 逐元素相乘，门决定每一维通过多少
y = g·W_downᵀ :: [B, S, D] :: 下投影回残差流维度
$ d_ff 取 8/3·D，3·D·d_ff ≈ 8D²，与标准 FFN 参数量持平
> SiLU(x) = x·σ(x)：σ 是 0~1 的开关，x 是内容，所以叫「自带门控」
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8/3 * d_model)  # 保持参数量与标准FFN相当
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """x: [B, S, D]"""
        gate = F.silu(self.w_gate(x))   # 门控分支
        up = self.w_up(x)               # 值分支
        return self.w_down(gate * up)    # 门控相乘 → 下投影

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    swiglu = SwiGLU(D)
    out = swiglu(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 512]
    print(f"Output shape: {out.shape}") # [2, 10, 512]
```

## 🎯 面试要点

- **三个权重矩阵** (Gate/Up/Down)，标准 FFN 只有两个
- **SiLU(x) = x × sigmoid(x)**，也称 Swish
- **d_ff 通常取 8/3 × d_model**（保持总参数量相当）
- **LLaMA, PaLM, Mistral, Qwen 等主流模型都使用**
- **面试常问**: SwiGLU 和标准 FFN 的区别？为什么效果更好？


