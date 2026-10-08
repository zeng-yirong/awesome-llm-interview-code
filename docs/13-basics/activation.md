# 激活函数 (Activation Functions)

> ReLU / GELU / SiLU 及其梯度

## 📌 原理与思想

激活函数引入非线性。ReLU 简单高效但有 dead neuron 问题；GELU 平滑更优（Transformer 常用）；SiLU/Swish 用于门控机制。

**为什么需要激活函数？**
- 引入非线性，使网络可以拟合复杂函数
- 没有激活函数：多层网络等价于单层
- 不同激活函数影响训练稳定性和效果

## 📐 核心公式

```
ReLU(x) = max(0, x)            ReLU'(x) = x > 0
GELU(x) = x · Φ(x)             Φ = standard normal CDF
SiLU(x) = x · σ(x)            SiLU'(x) = SiLU(x) + σ(x)(1-SiLU(x))
Swish = SiLU (same thing)

使用场景:
  ReLU:  标准 FFN (原始 Transformer)
  GELU:  BERT, GPT-2/3
  SiLU:  SwiGLU (LLaMA, PaLM)
```

## 📊 张量流程图

```
     ReLU              GELU              SiLU/Swish
  y │     ╱         y │        ╱      y │        ╱
    │    ╱             │      ╱         │      ╱
    │   ╱              │    ╱           │    ╱
    │──╱────── x        │──╱──────── x    │──╱──────── x
    │                   │  ╱ (平滑)       │ ╱ (门控)

梯度:
  ReLU':  [0, 0, 0, ..., 1, 1, 1]  (不连续)
  GELU':  平滑过渡
  SiLU':  平滑 + 门控
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


