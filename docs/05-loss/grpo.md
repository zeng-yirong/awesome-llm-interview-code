# GRPO 损失 (Group Relative Policy Optimization)

> 去掉 Critic，组内归一化优势，DeepSeek-R1

## 📌 原理与思想

PPO 的简化版：对同一问题生成 G 个回答，用组内归一化的奖励作为优势，不需要 Critic 网络。

**它解决什么问题**
- PPO 需要一个与策略同规模的 Critic（Value Head）估计状态价值，它自己也要训练调参，显存约占 40%，估不准还会把策略带偏。
- 数学题这类任务只在序列末尾给一个标量奖励（答案对/错），中间步骤的价值几乎学不出来，Critic 退化严重。
- 换个思路：同一个问题采样 G 个回答，它们的奖励天然可比 —— 组内均值是难度基线，组内标准差是区分度。

**核心思想**
- 把「这条回答好不好」换成「它比同组平均好多少」，Critic 被一组统计量取代。
- `Aᵢ = (rᵢ - mean) / (std + ε)`：减均值让优势有正有负，除标准差让不同难度题目的梯度量级一致。

**算法步骤与推导**
- 采样 G 个回答 → 各自打分 → 组内归一化得优势 → 用 PPO 的 clip 目标更新 → 加显式 KL 惩罚拉住参考策略。
- 全对/全错的组 std = 0，归一化后优势全 0，本来就没有梯度 —— DAPO 正是据此把它们直接丢弃。

**对比与代价**
- 相对 PPO：省掉 Critic 网络与价值损失，显存和调参成本都下降；代价是每组要多采 G 个回答。
- 相对 DPO：DPO 只吃静态偏好对、完全 off-policy；GRPO 是 on-policy，能在线探索，但要求可反复采样并打分。
- 三个弱点正是后三题的动机：token 级比率方差大（GSPO 改序列级）、全对全错组白算（DAPO 动态采样）、clip 上下界不对称（DAPO clip-higher）。

## 📐 核心公式

```
优势: Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)    # 组内归一化

损失: L = -E[min(ρᵢAᵢ, clip(ρᵢ)Aᵢ)] + β·KL(π‖π_ref)

ρᵢ = πθ(oᵢ|q) / πθ_old(oᵢ|q)

对比 PPO:
  PPO:  需要 Critic 估计 V(s) → A = r + γV - V
  GRPO: 组内归一化 → A = (r - mean) / std
```

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


