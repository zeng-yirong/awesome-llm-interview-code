# 广义优势估计 (GAE)

> 偏差-方差折衷的 λ-return 优势估计

## 📌 原理与思想

通过 λ 参数在蒙特卡洛（低偏差高方差）和 TD(0)（高偏差低方差）之间折衷。λ=1 等价于 MC，λ=0 等价于 TD(0)。

**它解决什么问题**
- MC return 直接取整条轨迹的回报：无偏差，但方差随序列长度增长，长轨迹上估计极不稳定。
- TD(0) 只往前看一步：方差小，但严重依赖 `V(s)` 的准确度，偏差大。
- 优势估计必须同时兼顾这两端，而纯 MC 和纯 TD 都停在极端上。

**核心思想**
- 不只取端点，而是把从 `t` 往后的所有 `k` 步 TD 误差按 `(γλ)^k` 加权求和。
- `λ` 就是权重衰减率：`λ → 0` 只剩最近一步（退化成 TD），`λ → 1` 权重均匀，等价于 MC。
- 指数衰减让越远的时间步影响越小，正好对上「越远越不确定」的直觉。

**算法步骤与推导**
- TD 误差 `δ_t = r_t + γ·V(s_{t+1}) - V(s_t)`，是 GAE 的原子单元。
- `A_t = Σ_l (γλ)^l · δ_{t+l}`，即各阶 TD 误差的几何加权和。
- 展开成 `A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + …`，`λ = 1` 时望远镜式相消，退化成 `MC return - V(s_t)`。
- 实现上不必真的算这个级数，用反向递推一步到位：`A_t = δ_t + γλ·A_{t+1}`，从 `A_T = δ_T` 往前推。

**对比与代价**
- 相对 MC：方差显著变小，长轨迹上训练稳定得多；相对 TD(0)：偏差更小，对 `V` 的估计误差没那么敏感。
- 代价是多了一个 `λ` 超参，通常取 0.95 配合 `γ = 0.99`，等于「几乎 MC 但略带衰减」。
- 它仍然依赖 critic 的 `V(s)`，critic 不准时 GAE 的偏差整体偏高 —— 这也是 GRPO 干脆去掉 critic 的动机之一。

## 📐 核心公式

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^{l}\,\delta_{t+l}
=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^{2}\delta_{t+2}+\cdots
$$

$$
A_t=\delta_t+\gamma\lambda\,A_{t+1},\qquad A_T=\delta_T
$$

$$
\begin{aligned}
\lambda=1: &\quad A_t=\text{MC return}-V(s_t) & \text{(unbiased, high variance)}\\[2pt]
\lambda=0: &\quad A_t=\delta_t & \text{(low variance, high bias)}
\end{aligned}
$$

$$
\gamma=0.99,\qquad \lambda=0.95
$$

## 📊 张量流程图

```
# 两个端点
! ✗ MC return：无偏差，但方差随轨迹长度增长
! ✗ TD(0)：方差小，但严重依赖 V(s) 的准确度，偏差大
# GAE：各阶 TD 误差的几何加权和
δ_t = r_t + γ·V(s_{t+1}) - V(s_t) :: TD 误差，GAE 的原子单元
A_t = Σ_l (γλ)^l · δ_{t+l} :: 从 t 往后所有 TD 误差按 (γλ)^l 加权
A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + … :: 展开形式
# 反向递推实现
A_T = δ_T :: 从末端起步
A_t = δ_t + γλ·A_{t+1} :: 往前推一步，不必真的算级数
$ λ=1 望远镜式相消 → MC return - V(s_t)；λ=0 → 只剩 δ_t，退化成 TD(0)
> 通常取 γ=0.99、λ=0.95，等于「几乎 MC 但略带衰减」
```

## 💻 代码实现

```python
import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    计算 GAE 优势
    
    rewards: [T]  每步奖励
    values:  [T+1] 价值估计 (含最后一步)
    
    返回: advantages [T], returns [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae       # 递推!
        advantages[t] = gae

    returns = advantages + values[:-1]         # returns = A + V
    return advantages, returns

# 测试
if __name__ == "__main__":
    T = 10
    rewards = torch.randn(T)
    values = torch.randn(T + 1)
    
    advantages, returns = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    print(f"Advantages shape: {advantages.shape}")  # [10]
    print(f"Returns shape: {returns.shape}")        # [10]
    
    # 验证: returns = advantages + values[:-1]
    assert torch.allclose(returns, advantages + values[:-1])

# PPO 中使用
def ppo_step(model, data, gamma=0.99, lam=0.95):
    rewards = data['rewards']
    values = model.compute_values(data['states'])
    advantages, returns = compute_gae(rewards, values, gamma, lam)
    
    # 归一化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO 更新
    old_logp = model.get_logprobs(data['states'], data['actions'])
    for _ in range(4):  # PPO epochs
        new_logp = model.get_logprobs(data['states'], data['actions'])
        loss = ppo_loss(old_logp, new_logp, advantages)
        loss.backward()
```

## 🎯 面试要点

- **λ 控制偏差-方差折衷**: λ=1 无偏差高方差, λ=0 高偏差低方差
- **递推实现 O(T)**: A_t = δ_t + γλ·A_{t+1}
- **通常 γ=0.99, λ=0.95**
- **PPO 训练的标准优势估计方法**
- **GRPO 不需要 GAE** (用组内归一化代替)
- **面试常问**: GAE 的 λ 参数有什么作用？


