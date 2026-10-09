# PPO 损失 (Proximal Policy Optimization)

> 截断重要性采样比率，RLHF 核心

## 📌 原理与思想

### 核心概念
通过截断重要性采样比率 r_t = π_new/π_old 到 [1-ε, 1+ε]，限制策略更新幅度，防止策略崩溃。相比普通策略梯度容易更新过大导致崩溃，PPO 通过 clip 机制保证训练稳定性，是 ChatGPT/InstructGPT 的 RLHF 核心算法。

### 核心思想
重要性采样比率 r_t 衡量新旧策略的差异。通过 clip 将 r_t 限制在 [1-ε, 1+ε] 范围内，当 A>0 时防止 ratio 过大（过度奖励），当 A<0 时防止 ratio 过小（过度惩罚），实现保守更新。

### 算法步骤
1. 计算重要性采样比率：ratio = exp(new_logp - old_logp)
2. 截断比率：clipped = clamp(ratio, 1-ε, 1+ε)
3. 计算未截断代理损失：surr1 = ratio × advantages
4. 计算截断代理损失：surr2 = clipped × advantages
5. 取较小值：loss = -mean(min(surr1, surr2))

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
# 截断重要性比率，给更新幅度设上界
old_logp :: π_old 对已采样动作的对数概率
new_logp :: π_new 的对数概率，需要梯度
ratio = exp(new_logp - old_logp) :: 重要性采样比率，用对数相减更稳
+ unclipped :: ratio · A :: 不加约束的更新量
+ clipped :: clip(ratio, 1-ε, 1+ε) · A :: 比率被夹住后的更新量
loss = -mean(min(unclipped, clipped)) :: 取更小的那个，得到悲观下界
! A > 0 时 ratio 涨过 1+ε 停止奖励，防过度优化
! A < 0 时 ratio 跌破 1-ε 停止惩罚，防过度惩罚
$ ε 通常取 0.2，一步更新幅度便有了显式上界
> ratio 必须用旧策略的 logp 现算，rollout 与更新要严格配对
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


