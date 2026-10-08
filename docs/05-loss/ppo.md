# PPO 损失 (Proximal Policy Optimization)

> 截断重要性采样比率，RLHF 核心

## 📌 原理与思想

通过截断重要性采样比率 r_t = π_new/π_old 到 [1-ε, 1+ε]，限制策略更新幅度，防止策略崩溃。

**为什么需要 PPO？**
- 策略梯度方法容易更新过大导致崩溃
- PPO 通过 clip 限制更新幅度
- ChatGPT/InstructGPT 的 RLHF 核心算法

## 📐 核心公式

```
L_PPO = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]

r_t = exp(log_π_new - log_π_old)
A_t: 优势函数 (GAE 估计)
ε: 截断参数，通常 0.2

裁剪机制:
- A > 0 (好动作): ratio > 1+ε 时停止奖励 (防过度优化)
- A < 0 (坏动作): ratio < 1-ε 时停止惩罚 (防过度惩罚)
```

## 📊 张量流程图

```
ratio = exp(new_logp - old_logp)    # 重要性采样比率
  │
  ├── unclipped = ratio × advantage
  ├── clipped   = clamp(ratio, 1-ε, 1+ε) × advantage
  │
  └── loss = -mean(min(unclipped, clipped))

当 A > 0 (好动作): ratio > 1+ε 时停止奖励 (防过度优化)
当 A < 0 (坏动作): ratio < 1-ε 时停止惩罚 (防过度惩罚)
```

## 💻 代码实现

```python
import torch

def ppo_loss(old_logp, new_logp, advantages, eps=0.2):
    """
    PPO 截断损失
    
    old_logp, new_logp: [B]  对数概率
    advantages: [B]  优势估计 (GAE)
    """
    ratio = torch.exp(new_logp - old_logp)           # r_t
    clipped = torch.clamp(ratio, 1-eps, 1+eps)       # clip(r_t)

    surr1 = ratio * advantages                        # 未截断
    surr2 = clipped * advantages                      # 截断

    return -torch.min(surr1, surr2).mean()            # 保守更新

# 完整 PPO 训练流程
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, critic_model):
        self.policy = policy_model
        self.ref = ref_model
        self.reward = reward_model
        self.critic = critic_model
    
    def train_step(self, prompts, responses):
        # 1. 计算 reward
        rewards = self.reward(prompts, responses)
        
        # 2. 计算 value 和 advantage
        values = self.critic(prompts, responses)
        advantages = compute_gae(rewards, values)
        
        # 3. 计算 log probs
        old_logp = self.policy.get_logprobs(prompts, responses)
        
        # 4. PPO 更新
        for _ in range(ppo_epochs):
            new_logp = self.policy.get_logprobs(prompts, responses)
            loss = ppo_loss(old_logp, new_logp, advantages)
            
            # KL penalty (防止偏离参考模型太远)
            ref_logp = self.ref.get_logprobs(prompts, responses)
            kl = (new_logp - ref_logp).mean()
            loss += kl_coef * kl
            
            loss.backward()
            optimizer.step()

# 测试
if __name__ == "__main__":
    B = 32
    old_logp = torch.randn(B)
    new_logp = old_logp + torch.randn(B) * 0.1
    advantages = torch.randn(B)
    
    loss = ppo_loss(old_logp, new_logp, advantages, eps=0.2)
    print(f"PPO Loss: {loss.item():.4f}")
```

## 🎯 面试要点

- **ratio = exp(new_logp - old_logp)** 重要性采样比率
- **clip 到 [1-ε, 1+ε]** 限制更新幅度
- **A>0**: 防止 ratio 过大 → 停止奖励
- **A<0**: 防止 ratio 过小 → 停止惩罚
- **ChatGPT/InstructGPT 的 RLHF 核心算法**
- **需要 reward model + critic model + reference model**
- **面试常问**: PPO 的 clip 机制是如何工作的？


