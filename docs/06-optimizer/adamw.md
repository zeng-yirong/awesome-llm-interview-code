# AdamW 优化器

> 解耦权重衰减，LLM 训练标配

## 📌 原理与思想

### 核心概念
Adam 的改进版：将权重衰减从梯度中解耦，直接作用于参数。相比 Adam 的权重衰减作用在梯度上正则化效果差，AdamW 解耦后正则化效果更好，是 LLM 训练的标准优化器。

### 核心思想
维护一阶矩（动量）和二阶矩（未中心化的方差），通过偏差修正确保初始阶段的稳定性。权重衰减直接作用于参数而非梯度，实现解耦正则化。

### 算法步骤
1. 更新一阶矩：m_t = β₁·m + (1-β₁)·g
2. 更新二阶矩：v_t = β₂·v + (1-β₂)·g²
3. 偏差修正：m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
4. 解耦权重衰减：θ = θ - lr·λ·θ
5. 参数更新：θ = θ - lr·m̂/(√v̂ + ε)

## 📐 核心公式

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1}+(1-\beta_1)\,g_t\\[2pt]
v_t &= \beta_2 v_{t-1}+(1-\beta_2)\,g_t^{2}\\[2pt]
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^{t}}\\[2pt]
\theta_t &= \theta_{t-1}-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\,\theta_{t-1}\right)
\end{aligned}
$$

$$
\eta=3\times10^{-4},\qquad (\beta_1,\beta_2)=(0.9,\ 0.95),\qquad \lambda=0.1,\qquad \epsilon=10^{-8}
$$

## 📊 张量流程图

```
# 一阶矩与二阶矩
g_t :: 当前梯度
m_t = β₁·m + (1-β₁)·g_t :: 一阶矩，动量
v_t = β₂·v + (1-β₂)·g_t² :: 二阶矩，梯度平方的滑动平均
# 偏差修正
m̂ = m / (1-β₁ᵗ) :: 补偿前几步从 0 起步导致的偏小
v̂ = v / (1-β₂ᵗ)
# 更新：自适应梯度步 + 解耦的权重衰减
θ = θ - lr·(m̂/(√v̂ + ε) + λ·θ) :: 两项相加，衰减项不经过任何归一化
$ Adam 把 λθ 加进梯度，再被 1/√v̂ 归一化，衰减强度因此失控
> LLM 标配：lr = 3e-4、betas = (0.9, 0.95)、weight_decay = 0.1
```

## 💻 代码实现

```python
import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.eps = lr, eps
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            
            # 解耦权重衰减 (AdamW vs Adam 的关键区别)
            if self.wd != 0:
                p.data.mul_(1 - self.lr * self.wd)
            
            # 更新矩
            self.m[i] = self.b1 * self.m[i] + (1-self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * g**2
            
            # 偏差修正 + 更新
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p.data.add_(-self.lr * m_hat / (v_hat.sqrt() + self.eps))

# 学习率调度 (LLM 标配)
def cosine_schedule_with_warmup(step, warmup_steps, total_steps, max_lr=3e-4):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

# 测试
if __name__ == "__main__":
    model = torch.nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    x = torch.randn(4, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    print("AdamW step completed")
```

## 🎯 面试要点

- **AdamW vs Adam**: 权重衰减直接作用于参数，非梯度
- **偏差修正**解决初始阶段 m,v 偏小的问题
- **LLM 标配**: lr=3e-4, betas=(0.9,0.95), wd=0.1
- **配合 cosine schedule with warmup** 使用
- **面试常问**: AdamW 和 Adam 的区别？为什么要解耦权重衰减？


