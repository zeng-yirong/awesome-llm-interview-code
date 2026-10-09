# 梯度与反向传播 (Gradient & Backpropagation)

> 链式法则，深度学习的基石

## 📌 原理与思想

反向传播利用链式法则从输出向输入逐层计算梯度。计算图的前向传播保存中间变量，反向传播利用这些变量计算梯度。

**它解决什么问题**
- 手推每层的梯度不现实：参数量上亿，每改一次结构就要重新推导一遍。
- 数值微分（逐参数扰动）要对每个参数各跑一次前向，`n` 个参数就是 `n` 次前向，完全不可行。
- 需要一种「一次前向 + 一次反向就把所有参数的梯度全拿到」的方法。

**核心思想**
- 把整个网络看成一个计算图，每个算子只负责「给定输出的梯度，算出输入的梯度」这一件事。
- 链式法则保证：只要能从输出往输入逐层把局部梯度乘起来，就得到了每个参数的梯度。
- 前向保存的中间量（激活、输入）在反向时就是现成的乘数，省掉大量重算 —— 这是它高效的关键。

**算法步骤与推导**
- 前向按拓扑顺序算下去，沿途把 `x`、`h`、`a`、`ŷ` 等中间量存下来。
- 反向从 `∂L/∂ŷ` 起步逐层回传：`∂L/∂W₂ = ∂L/∂ŷ·aᵀ`、`∂L/∂a = W₂ᵀ·∂L/∂ŷ`。
- 过 `ReLU` 时梯度乘以 `(a > 0)`，负区间直接归零；再往前 `∂L/∂W₁ = ∂L/∂h·xᵀ`、`∂L/∂x = W₁ᵀ·∂L/∂h`。
- 每个参数在轮到它的那一刻就拿到梯度，不需要额外前向次数；复杂度与前向同阶。

**对比与代价**
- 相对数值微分：一次反向拿到全部梯度，代价与前向同阶，而不是随参数量线性增长。
- 代价是中间激活要一直留到反向，显存随层数线性增长 —— 这正是梯度检查点要解决的问题。
- 代价是反向依赖前向的完整计算图，动态控制流（`.item()`、原地修改）会破坏图，是常见的一类训练 bug。

## 📐 核心公式

$$
\frac{\partial\mathcal{L}}{\partial x}=\frac{\partial\mathcal{L}}{\partial y}\cdot\frac{\partial y}{\partial x}
$$

$$
\begin{aligned}
y=Wx: &\quad \frac{\partial\mathcal{L}}{\partial W}=\frac{\partial\mathcal{L}}{\partial y}\,x^{\top},\quad
\frac{\partial\mathcal{L}}{\partial x}=W^{\top}\frac{\partial\mathcal{L}}{\partial y}\\[3pt]
y=\operatorname{ReLU}(x): &\quad \frac{\partial\mathcal{L}}{\partial x}=\frac{\partial\mathcal{L}}{\partial y}\odot(x>0)\\[3pt]
y=\operatorname{softmax}(z): &\quad \frac{\partial\mathcal{L}}{\partial z}=y-\text{one\_hot}(\text{target})\quad(\text{with CE})
\end{aligned}
$$

## 📊 张量流程图

```
# 前向：按拓扑顺序算，顺手保存中间量
x → W₁ → h → ReLU → a → W₂ → ŷ → Loss → L :: 保存 x、h、a、ŷ 供反向使用
# 反向：从输出往输入逐层乘局部梯度
∂L/∂ŷ :: 从 Loss 的导数起步
∂L/∂W₂ = ∂L/∂ŷ · aᵀ :: 权重梯度只需输出的梯度和该层输入
∂L/∂a = W₂ᵀ · ∂L/∂ŷ :: 继续往左传
∂L/∂h = ∂L/∂a · (a>0) :: ReLU 的局部导数，负区间直接归零
∂L/∂W₁ = ∂L/∂h · xᵀ :: 每层模式完全相同，逐层套用
$ 一次前向 + 一次反向 = 全部参数的梯度，复杂度与前向同阶
> 代价是中间激活要留到反向，显存随层数线性增长
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


