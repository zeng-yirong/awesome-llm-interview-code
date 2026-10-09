# GRPO 损失 (Group Relative Policy Optimization)

> 去掉 Critic，组内归一化优势，DeepSeek-R1

## 📌 原理与思想

### 核心概念
PPO 的简化版：对同一问题生成 G 个回答，用组内归一化的奖励作为优势，不需要 Critic 网络。相比 PPO 节省约 40% 训练显存，是 DeepSeek-R1 使用的强化学习算法。

### 核心思想
对同一问题生成 G 个回答，用组内归一化的奖励代替 Critic 网络的价值估计。优势函数 Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)，结合 PPO clip 机制和显式 KL 惩罚。

### 算法步骤
1. 对同一问题 q 生成 G 个回答：o₁, o₂, ..., o_G
2. 计算每个回答的奖励：r₁, r₂, ..., r_G
3. 组内归一化优势：Aᵢ = (rᵢ - mean) / (std + ε)
4. 计算 PPO clip 损失
5. 添加显式 KL 惩罚：loss += β · KL(π‖π_ref)

## 📐 核心公式

$$
\hat{A}_i=\frac{r_i-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})+\epsilon},
\qquad
\{o_1,\dots,o_G\}\sim\pi_{\theta_{\text{old}}}(\cdot\mid q)
$$

$$
\mathcal{L}(\theta)=-\mathbb{E}\!\left[\min\!\left(\rho_i\hat{A}_i,\ \operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\,\hat{A}_i\right)\right]
+\beta\,\mathrm{KL}\!\left(\pi_\theta\,\|\,\pi_{\text{ref}}\right)
$$

$$
\begin{aligned}
\text{GRPO (sequence-level):}\quad & \rho_i=\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}\\
\text{PPO (token-level):}\quad & \rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid x,y_{<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid x,y_{<t})}
\end{aligned}
$$

## 📊 张量流程图

```
# 组内采样：同一个问题采 G 个回答
问题 q :: 同一个 prompt
+ 回答 o₁ :: 奖励 r₁
+ 回答 o₂ :: 奖励 r₂
+ 回答 o_G :: 奖励 r_G（共 G 个，共享同一个 mean/std）
组内归一化 :: Aᵢ = (rᵢ - mean) / (std + ε) :: 有正有负才有梯度方向
$ 组内统计量代替 Critic：整个训练不需要 Value Head

# 策略更新（PPO 的 clip 目标）
比率 ρᵢ :: πθ(oᵢ|q) / πθ_old(oᵢ|q)
做 clip :: clip(ρᵢ, 1-ε, 1+ε) :: 限制单步更新幅度
$ L = -E[min(ρᵢAᵢ, clip(ρᵢ)·Aᵢ)] + β·KL(π‖π_ref)
> 显式 KL 惩罚拉住参考策略，防止跑偏

# 与 PPO 的对比
> PPO:  需要 Critic 估计 V(s) → A = r + γV - V
> GRPO: 组内归一化 → A = (r - mean) / std，省掉 Critic
```

## 💻 代码实现

```python
import torch

def grpo_advantages(rewards):
    """组内归一化优势 (不需要 Critic)"""
    # rewards: [batch, group_size]
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_loss(old_logp, new_logp, advantages, eps=0.2, beta=0.01, ref_kl=None):
    """
    GRPO 损失 = PPO clip loss + β * KL penalty
    """
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1-eps, 1+eps)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    if ref_kl is not None:
        loss = loss + beta * ref_kl       # 显式 KL 惩罚
    return loss.mean()

def kl_penalty(logp, ref_logp):
    """Schulman KL 估计器 (低方差)"""
    ratio = torch.exp(ref_logp - logp)
    return (ratio - (ref_logp - logp) - 1).mean()

# 完整 GRPO 训练流程
class GRPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, group_size=8):
        self.policy = policy_model
        self.ref = ref_model
        self.reward = reward_model
        self.group_size = group_size
    
    def train_step(self, prompts):
        # 1. 对每个 prompt 生成 group_size 个回答
        responses = []
        for _ in range(self.group_size):
            responses.append(self.policy.generate(prompts))
        
        # 2. 计算每个回答的 reward
        rewards = torch.stack([
            self.reward(prompts, resp) for resp in responses
        ], dim=1)  # [batch, group_size]
        
        # 3. 组内归一化优势
        advantages = grpo_advantages(rewards)
        
        # 4. GRPO 更新
        for i in range(self.group_size):
            old_logp = self.policy.get_logprobs(prompts, responses[i])
            new_logp = self.policy.get_logprobs(prompts, responses[i])
            ref_logp = self.ref.get_logprobs(prompts, responses[i])
            
            kl = kl_penalty(new_logp, ref_logp)
            loss = grpo_loss(old_logp, new_logp, advantages[:, i],
                           ref_kl=kl)
            loss.backward()

# 测试
if __name__ == "__main__":
    # 组内归一化
    rewards = torch.tensor([[1.0, 2.0, 3.0, 0.5],
                            [2.0, 2.5, 1.5, 3.0]])
    advantages = grpo_advantages(rewards)
    print("Advantages:", advantages)
    # 每行均值为 0，标准差为 1
```

## 🎯 面试要点

- **核心**: 组内归一化代替 Critic 网络
- **对同一问题生成 G 个回答**，计算组内相对优势
- **不需要 Value Head**，节省约 40% 训练显存
- **显式 KL 惩罚**防止偏离参考策略太远
- **DeepSeek-R1 使用 GRPO** 进行强化学习
- **面试常问**: GRPO 和 PPO 的区别？为什么不需要 Critic？


