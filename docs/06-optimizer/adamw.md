# AdamW 优化器

> 解耦权重衰减，LLM 训练标配

## 📌 原理与思想

Adam 的改进版：将权重衰减从梯度中解耦，直接作用于参数。正则化效果更好，是 LLM 训练的标准优化器。

**为什么用 AdamW？**
- Adam 的权重衰减作用在梯度上，正则化效果差
- AdamW 将权重衰减解耦，直接作用于参数
- LLM 训练的标准优化器

## 📐 核心公式

```
m_t = β₁m_{t-1} + (1-β₁)g_t          # 一阶矩
v_t = β₂v_{t-1} + (1-β₂)g_t²         # 二阶矩
m̂_t = m_t/(1-β₁ᵗ)                     # 偏差修正
v̂_t = v_t/(1-β₂ᵗ)

θ_t = θ_{t-1} - lr · (m̂_t/(√v̂_t + ε) + λθ_{t-1})
                                ↑ 解耦权重衰减

LLM 标配超参:
- lr = 3e-4
- betas = (0.9, 0.95)
- weight_decay = 0.1
```

## 📊 张量流程图

```
gradient g_t
  │
  ├── m_t = β₁·m + (1-β₁)·g       # 一阶矩 (动量)
  ├── v_t = β₂·v + (1-β₂)·g²      # 二阶矩
  │
  ├── m̂ = m/(1-β₁ᵗ)                # 偏差修正
  ├── v̂ = v/(1-β₂ᵗ)
  │
  └── θ = θ - lr·(m̂/(√v̂+ε) + λ·θ) # 更新 + 解耦权重衰减
                       ↑ adaptive    ↑ decoupled wd
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


