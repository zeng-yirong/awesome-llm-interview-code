# 激活函数 (Activation Functions)

> ReLU / GELU / SiLU 及其梯度

## 📌 原理与思想

激活函数引入非线性。ReLU 简单高效但有 dead neuron 问题；GELU 平滑更优（Transformer 常用）；SiLU/Swish 用于门控机制。

**它解决什么问题**
- 没有激活函数时，多层线性变换叠起来仍然等价于单层线性变换，深度白给。
- 反向传播要求处处可导或几乎处处可导，`sign`、`step` 这类硬阈值函数没法直接用在深层网络里。
- 不同激活的饱和行为差别很大，直接决定训练是否稳定。

**核心思想**
- 用一个逐元素的非线性函数把线性层的结果掰弯，让「多层」真正带来表达力。
- 现代选择集中在平滑版 ReLU 上：负区间给一点非零梯度，避免神经元被永久关死。
- 门控形式的 `x·σ(x)` 顺带把「通过多少」也变成可学的，这正是 SwiGLU 的基础。

**算法步骤与推导**
- `ReLU(x) = max(0, x)`：正区间导数恒为 1，负区间恒为 0。
- `GELU(x) = x·Φ(x)`（`Φ` 是标准正态 CDF）：负区间仍有小梯度，是 ReLU 的平滑版。
- `SiLU(x) = x·σ(x)`：`σ` 充当 0~1 的开关，负区间先降后回升，自带门控。
- 三者都是逐元素作用，形状完全不变，可以随意替换。

**对比与代价**
- `ReLU` 实现最便宜，但负区间梯度恒为 0，神经元长期落在负区间就再也学不动（dead neuron）。
- `GELU`、`SiLU` 平滑、处处可导，训练更稳，代价是都要算 exp 或 erf，比 `max` 贵一些。
- 目前标准 FFN 多用 `GELU`（BERT、GPT-2/3），门控 FFN 用 `SiLU`（LLaMA、PaLM）。

## 📐 核心公式

$$
\operatorname{ReLU}(x)=\max(0,x),\qquad \operatorname{ReLU}^{\prime}(x)=(x>0)
$$

$$
\operatorname{GELU}(x)=x\,\Phi(x),\qquad \Phi=\text{standard normal CDF}
$$

$$
\operatorname{SiLU}(x)=x\,\sigma(x),\qquad
\operatorname{SiLU}^{\prime}(x)=\operatorname{SiLU}(x)+\sigma(x)\left(1-\operatorname{SiLU}(x)\right)
$$

$$
\text{Swish}\equiv\operatorname{SiLU}
$$

## 📊 张量流程图

```
# 三种激活函数：形状决定行为
x :: [..., D] :: 输入张量
+ ReLU :: max(0, x) :: 负区间恒为 0，正区间线性
+ GELU :: x·Φ(x) :: 平滑版 ReLU，负区间仍有小梯度
+ SiLU :: x·σ(x) :: 自带门控，负区间先降后回升
y :: [..., D] :: 逐个元素作用，形状不变

# 梯度
+ ReLU' :: x > 0 :: 只有 0 / 1 两档，不连续
+ GELU' :: 平滑过渡 :: 处处可导
+ SiLU' :: 平滑 + 门控 :: 处处可导

# 使用场景
+ ReLU :: 原始 Transformer 的标准 FFN
+ GELU :: BERT、GPT-2/3
+ SiLU :: SwiGLU 门控（LLaMA、PaLM）
> ReLU 负区间梯度为 0，神经元长期落在负区间就再也学不动（dead neuron）
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F

# ReLU
def relu(x):
    return torch.clamp(x, min=0)    # 或 F.relu(x)

# GELU (近似版本更快)
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        (2/3.14159)**0.5 * (x + 0.044715 * x**3)
    ))
    # 或直接用 F.gelu(x)

# SiLU (Swish)
def silu(x):
    return x * torch.sigmoid(x)     # 或 F.silu(x)

# 梯度对比
def check_gradients():
    x = torch.linspace(-3, 3, 100, requires_grad=True)
    
    # ReLU
    y_relu = relu(x).sum()
    y_relu.backward()
    print("ReLU grad:", x.grad[:5])     # [0, 0, 0, ..., 1, 1]
    x.grad.zero_()
    
    # GELU
    y_gelu = gelu(x).sum()
    y_gelu.backward()
    print("GELU grad:", x.grad[:5])     # 平滑过渡
    x.grad.zero_()
    
    # SiLU
    y_silu = silu(x).sum()
    y_silu.backward()
    print("SiLU grad:", x.grad[:5])     # 平滑 + 门控

# 在 Transformer 中使用
class FFN_ReLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class FFN_GELU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

# 测试
if __name__ == "__main__":
    check_gradients()
    
    # 可视化
    import matplotlib.pyplot as plt
    
    x = torch.linspace(-3, 3, 100)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x.numpy(), relu(x).detach().numpy())
    plt.title('ReLU')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x.numpy(), gelu(x).detach().numpy())
    plt.title('GELU')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(x.numpy(), silu(x).detach().numpy())
    plt.title('SiLU/Swish')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png')
    print("Plot saved to activation_functions.png")
```

## 🎯 面试要点

- **ReLU**: 简单高效，但有 dead neuron (负区间梯度=0)
- **GELU**: 平滑版 ReLU，BERT/GPT 使用
- **SiLU = Swish**: 用于 SwiGLU 门控机制
- **面试常问**: 各激活函数的梯度和使用场景？
- **对比**: ReLU vs GELU vs SiLU 的优缺点？


