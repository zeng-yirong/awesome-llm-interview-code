# PPO 损失 (Proximal Policy Optimization)

> 截断重要性采样比率，RLHF 核心

## 📌 原理与思想

通过截断重要性采样比率 r_t = π_new/π_old 到 [1-ε, 1+ε]，限制策略更新幅度，防止策略崩溃。

**它解决什么问题**
- 策略梯度是 on-policy 的：采一批数据更新一次就得丢掉，样本效率极低。
- 想拿同一批数据多更新几步，就得用重要性采样；但比率一旦偏离 1 太远，估计的方差会爆炸。
- 朴素的策略梯度没有约束，一步更新过大就会把策略推到一个再也回不来的坏区域，训练直接崩。

**核心思想**
- 允许策略偏离旧策略，但把重要性比率 `r_t` 限制在 `[1-ε, 1+ε]` 内，超出的部分不再提供梯度。
- 取 `min(未裁剪, 裁剪)` 而不是直接裁剪，是为了拿到悲观下界：真正生效的是两者中更小的那个。
- 于是「方向对但步子太大」的更新会被自动刹车，既保住样本效率又不至于崩。

**算法步骤与推导**
- `ratio = exp(new_logp - old_logp)`，用对数概率相减再取指数，比直接做除法数值稳定。
- 算两支：`unclipped = ratio·A`、`clipped = clamp(ratio, 1-ε, 1+ε)·A`，取两者逐元素的最小值。
- `A > 0` 时 `ratio` 涨过 `1+ε` 就被截住，防止一个本来就不错的动作被过度奖励；`A < 0` 时 `ratio` 跌到 `1-ε` 以下也被截住，防止过度惩罚。
- `ε` 通常取 0.2；取负号后求均值即为 loss，min 的悲观特性保证更新幅度有上界。

**对比与代价**
- 相对朴素策略梯度：每一步的更新幅度有显式上界，训练稳得多；相对 TRPO：用一次 clip 代替二阶约束求解，实现简单很多。
- 代价是它只约束了比率的上界，对策略的长期漂移没有约束 —— RLHF 里还得额外加一项 KL 惩罚拉住参考模型。
- `ratio` 必须用旧策略的 logp 现算，所以 rollout 与更新之间要严格配对，工程上比 off-policy 方法麻烦。

## 📐 核心公式

$$
\mathcal{L}_{\text{PPO}}=-\mathbb{E}_t\left[\min\!\left(r_tA_t,\ \operatorname{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,A_t\right)\right]
$$

$$
r_t=\exp\!\left(\log\pi_\theta^{\text{new}}-\log\pi_\theta^{\text{old}}\right),
\qquad
A_t=\text{GAE advantage},
\qquad
\epsilon=0.2
$$

$$
\begin{aligned}
A_t>0:\ &\quad r_t>1+\epsilon\ \Rightarrow\ r_tA_t\ \text{capped at}\ (1+\epsilon)A_t\\[2pt]
A_t<0:\ &\quad r_t<1-\epsilon\ \Rightarrow\ r_tA_t\ \text{floored at}\ (1-\epsilon)A_t
\end{aligned}
$$

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


