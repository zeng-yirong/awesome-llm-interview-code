# DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)

> 四处改动修 GRPO：解耦裁剪、动态采样、token 级损失、超长惩罚

## 📌 原理与思想

### 核心概念
DAPO 不改 GRPO 的骨架，只针对四个已知缺陷动手：(1) clip-higher 解耦裁剪上下界，放开低概率 token 的上升空间以维持熵；(2) 动态采样过滤掉全对/全错的组，只留有梯度信号的组；(3) 用 token 级损失替代序列级平均，让长回答的每个 token 权重一致；(4) 超长奖励塑形，惩罚被截断的超长回答。四个改动互不耦合，最终在长链推理任务上明显更强；代价是超参变多（多一个 `ε_high`、一个 `α`、一个 `L_max`），动态采样在有效组不足时还要额外的重采样逻辑。

### 核心思想
不改 GRPO 的骨架，只针对上面四个已知缺陷定点修补。`clip-higher` 解耦上下界，把上界放得比下界宽，给低概率 token 留出上升通道以维持熵；动态采样过滤掉没有梯度信号的组；损失改为 token 级求和；再对超长回答做奖励塑形。

### 算法步骤
1. `clip-higher`：用 `clip(ρ, 1-ε_low, 1+ε_high)`，典型 `ε_low = 0.2`、`ε_high = 0.28`，抬高的是熵的下界。
2. 动态采样：丢掉 `Â` 全为 0 的组，只留组内奖励有正有负的组，等于把算力全花在有效样本上。
3. token 级损失：`L = -1/Σ_i|y_i| · Σ_i Σ_t min(...)`，分母是总 token 数而不是每组平均，长回答的每个 token 权重一致。
4. 超长奖励塑形：`R̃(y) = R(y) - α·max(0, |y| - L_max)`，超过上限就线性扣分。

## 📐 核心公式

```math
\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})},
\qquad
\hat{A}_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)+\epsilon}
```

```math
\text{clip-higher}:\quad
\min\!\left(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t},\,1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}})\,\hat{A}_i\right),
\qquad
\epsilon_{\text{low}}=0.2,\quad \epsilon_{\text{high}}=0.28
```

```math
\text{dynamic sampling}:\quad \text{discard groups with } \hat{A}_i\equiv 0
```

```math
\text{token-level loss}:\quad
\mathcal{L}=-\frac{1}{\sum_i|y_i|}\sum_i\sum_t\min\!\left(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t},\,1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}})\,\hat{A}_i\right)
```

```math
\text{overlong shaping}:\quad \tilde{R}(y)=R(y)-\alpha\max\!\left(0,\ |y|-L_{\max}\right)
```

## 📊 张量流程图

```
# 一个 batch 里的若干组，先过滤再更新
! ✗ group_1  rewards [1,1,1,1] → std = 0 → Â 全 0，无梯度信号
! ✗ group_2  rewards [0,0,0,0] → std = 0 → Â 全 0，无梯度信号
! ✓ group_3  rewards [1,0,1,0] → Â 有正有负，保留
> 动态采样：只走有梯度信号的组，全对全错的组等于白算
比率 ρ_{i,t} = exp(new_logp - old_logp) :: [B, G, T] :: 仍然是逐 token 的
clip-higher :: clip(ρ, 1-ε_low, 1+ε_high) :: ε_high > ε_low，给低概率 token 留上升空间
长度惩罚 :: R̃(y) = R(y) - α·max(0, |y| - L_max) :: 惩罚被截断的超长回答
loss = -Σ(保留组的全部 token) / Σ|y_i| :: 分母是总 token 数，不是每组平均
$ clip-higher 抬高的是熵的下界，缓解输出同质化
> 四个改动互不耦合，都不动 GRPO 的骨架
```

## 💻 代码实现

```python
import torch

def overlong_reward_shaping(rewards, lengths, max_len, alpha=1.0):
    """超长奖励塑形: 超出 max_len 的部分线性扣分"""
    over = (lengths - max_len).clamp_min(0).float()
    return rewards - alpha * over


def dynamic_sampling_filter(rewards, group_size=8):
    """
    动态采样: 丢掉组内奖励全相同的组（Â ≡ 0，没有梯度信号）
    rewards: [num_groups * group_size]  扁平化的组内奖励
    返回: 保留下来的组的索引
    """
    groups = rewards.view(-1, group_size)
    keep = groups.max(dim=-1).values != groups.min(dim=-1).values
    return keep.nonzero(as_tuple=True)[0]


def dapo_loss(per_token_logp_new, per_token_logp_old, completion_mask,
              advantages, eps_low=0.2, eps_high=0.28):
    """
    DAPO 损失: clip-higher + token 级归一化
    per_token_logp_*: [B, G, T]    advantages: [B, G]
    """
    # 1. token 级重要性比率
    ratio = (per_token_logp_new - per_token_logp_old).exp()      # [B, G, T]
    adv = advantages.unsqueeze(-1)                               # [B, G, 1]

    # 2. 解耦的上下界: 上界放松 → 低概率 token 有更大的上升空间
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    per_token = torch.min(ratio * adv, clipped * adv)            # [B, G, T]

    # 3. token 级归一化: 除以整个 batch 的总 token 数
    mask = completion_mask.float()
    return -(per_token * mask).sum() / mask.sum().clamp_min(1.0)


if __name__ == "__main__":
    # 动态采样: 前两组奖励全相同 → 丢弃
    rewards = torch.tensor([1., 1., 1., 1.,   0., 0., 0., 0.,   1., 0., 1., 0.,   1., 0., 0., 1.])
    print("保留的组:", dynamic_sampling_filter(rewards, group_size=4).tolist())

    # 超长塑形
    shaped = overlong_reward_shaping(
        torch.tensor([1.0, 1.0]), torch.tensor([100., 600.]), max_len=512, alpha=0.5)
    print("超长塑形后奖励:", shaped.tolist())

    # 主损失
    torch.manual_seed(0)
    B, G, T = 2, 4, 5
    old_logp = torch.randn(B, G, T) - 2.0
    new_logp = old_logp + 0.02 * torch.randn(B, G, T)
    mask = torch.ones(B, G, T)
    mask[:, :, -1] = 0
    adv = torch.randn(B, G)
    print("dapo loss:", dapo_loss(new_logp, old_logp, mask, adv).item())
```

## 🎯 面试要点

- **clip-higher 为什么有效**: 上界放松后，低概率 token 的比率有更大的上升空间，熵不会那么快坍缩；抬高的是熵的下界，不是直接把熵加进损失
- **动态采样为什么有效**: 全对/全错的组优势恒为 0，算力全白花；丢掉它们等于按「有没有梯度信号」筛数据
- **token 级归一化的影响**: 按序列长度归一化时，长回答的每个 token 梯度被稀释；除以总 token 数让每个 token 等权，长回答因此学得更好
- **超长惩罚是软约束**: 硬截断会让奖励突变、模型学不到收尾；线性/分段惩罚给的是平滑信号
- **四个改动的取向不同**: clip-higher 管探索（熵），动态采样管效率，token 级损失管长度偏置，超长塑形管长度控制
- **和 GSPO 的对照**: DAPO 保留 token 级比率（但改了归一化和裁剪界），GSPO 直接把比率提到序列级
- **面试常问**: DAPO 相对 GRPO 改了哪几处？各自的动机是什么？