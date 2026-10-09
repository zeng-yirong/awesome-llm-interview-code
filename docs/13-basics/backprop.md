# 梯度与反向传播 (Gradient & Backpropagation)

> 链式法则，深度学习的基石

## 📌 原理与思想

### 核心概念
反向传播利用链式法则从输出向输入逐层计算梯度。计算图的前向传播保存中间变量，反向传播利用这些变量计算梯度。相比手动推导梯度，反向传播自动计算且高效（一次前向 + 一次反向），是深度学习训练的基础。

### 核心思想
链式法则：∂L/∂x = ∂L/∂y · ∂y/∂x。前向传播时保存中间变量（计算图），反向传播时利用这些变量和链式法则逐层计算梯度。

### 算法步骤
1. 前向传播：逐层计算并保存中间变量
2. 计算损失：L = loss(y_pred, y_true)
3. 反向传播：从输出向输入逐层计算梯度
4. 常见梯度：y=Wx → ∂L/∂W = ∂L/∂y · xᵀ；ReLU → ∂L/∂x = ∂L/∂y · (x>0)
5. 参数更新：θ = θ - lr · ∂L/∂θ

## 📐 核心公式

```
链式法则: ∂L/∂x = ∂L/∂y · ∂y/∂x

常见梯度:
  y = Wx:     ∂L/∂W = ∂L/∂y · xᵀ,  ∂L/∂x = Wᵀ · ∂L/∂y
  y = ReLU(x): ∂L/∂x = ∂L/∂y · (x > 0)
  y = softmax: ∂L/∂z = y - one_hot(target)  (配合 CE)
```

## 📊 张量流程图

```
前向 (计算图):
  x → [W₁] → h → [ReLU] → a → [W₂] → ŷ → [Loss] → L
       保存 x,h,a,ŷ 用于反向

反向 (链式法则):
  ∂L/∂ŷ ← [Loss']
  ∂L/∂W₂ = ∂L/∂ŷ · aᵀ     ∂L/∂a = W₂ᵀ · ∂L/∂ŷ
  ∂L/∂h = ∂L/∂a · (a>0)   [ReLU']
  ∂L/∂W₁ = ∂L/∂h · xᵀ     ∂L/∂x = W₁ᵀ · ∂L/∂h
```

## 💻 代码实现

```python
import torch

# 手动实现简单两层网络的反向传播
class TwoLayerNet:
    def __init__(self, d_in, d_hidden, d_out):
        self.W1 = torch.randn(d_in, d_hidden) * 0.01
        self.W2 = torch.randn(d_hidden, d_out) * 0.01

    def forward(self, x):
        self.x = x
        self.h = x @ self.W1              # 线性
        self.a = torch.relu(self.h)        # 激活
        self.y = self.a @ self.W2          # 输出
        return self.y

    def backward(self, y_true, lr=0.01):
        # Loss = MSE = mean((y - y_true)²)
        dy = 2 * (self.y - y_true) / y_true.size(0)  # ∂L/∂y

        # ∂L/∂W2 = aᵀ · dy
        dW2 = self.a.T @ dy
        da = dy @ self.W2.T

        # ReLU 梯度
        dh = da * (self.h > 0).float()

        # ∂L/∂W1 = xᵀ · dh
        dW1 = self.x.T @ dh

        # 更新
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2

# 使用 PyTorch 自动求导
class TwoLayerNetAuto:
    def __init__(self, d_in, d_hidden, d_out):
        self.W1 = torch.randn(d_in, d_hidden, requires_grad=True) * 0.01
        self.W2 = torch.randn(d_hidden, d_out, requires_grad=True) * 0.01

    def forward(self, x):
        h = x @ self.W1
        a = torch.relu(h)
        y = a @ self.W2
        return y

# 测试
if __name__ == "__main__":
    # 手动反向传播
    net = TwoLayerNet(10, 20, 5)
    x = torch.randn(4, 10)
    y_true = torch.randn(4, 5)
    
    y_pred = net.forward(x)
    loss = ((y_pred - y_true) ** 2).mean()
    print(f"Loss before: {loss.item():.4f}")
    
    net.backward(y_true, lr=0.01)
    
    y_pred = net.forward(x)
    loss = ((y_pred - y_true) ** 2).mean()
    print(f"Loss after: {loss.item():.4f}")
    
    # PyTorch 自动求导
    net_auto = TwoLayerNetAuto(10, 20, 5)
    y_pred = net_auto.forward(x)
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()
    
    print(f"\nW1 grad shape: {net_auto.W1.grad.shape}")
    print(f"W2 grad shape: {net_auto.W2.grad.shape}")
```

## 🎯 面试要点

- **链式法则是反向传播的数学基础**
- **前向保存中间变量，反向利用它们计算梯度**
- **矩阵求导**: ∂(xW)/∂W = xᵀ · ∂L/∂y
- **ReLU 梯度**: 正区间为 1，负区间为 0
- **Softmax + CE 的梯度特别简洁**: y - one_hot
- **面试常问**: 反向传播的原理？链式法则如何应用？


