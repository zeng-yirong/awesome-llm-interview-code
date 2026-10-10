# 组序列策略优化 (Group Sequence Policy Optimization)

> GRPO 的比率提到序列级，MoE 训练的稳定解

## 📌 原理与思想

### 核心概念
GRPO 的重要性比率是 token 级的，每个 token 各自 clip；而奖励本身是序列级的（整条回答对错），粒度对不上。这种不匹配带来高方差：同一条序列里一部分 token 被裁掉、另一部分照常更新，序列内部的更新方向互相拉扯，在 MoE 模型上还会放大路由抖动导致训练发散。GSPO 把重要性比率定义在序列级 —— 对逐 token 对数比做长度归一化，整条序列共享一个标量比率、只 clip 一次；代价是粒度变粗，序列内部个别 token 的差异会被平均掉。

### 核心思想
既然奖励是序列级的一个标量，比率也应该是序列级的一个标量：整条序列共享同一个比率，只 clip 一次。把逐 token 的对数比先按 mask 求和、再除以序列长度，得到长度归一化的平均对数比；取指数就得到序列级比率 `s_i`，它衡量的是「整条回答在当前策略下比旧策略平均好多少」。

### 算法步骤
1. 组内优势 `Â_i` 与 GRPO 完全一致：`(r_i - mean(r)) / (std(r) + ε)`。
2. 逐 token 对数比 `[B, G, T]`，按 mask 求和压掉 padding → `[B, G]`，再除以 `|y_i|` 做长度归一化。
3. `s_i = exp(长度归一化的对数比)`，形状 `[B, G]` —— 每条序列一个标量，这正是「序列级」的含义。
4. 目标函数与 PPO 同形，只是把 `ρ` 换成 `s_i`：`min(s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i)`。

## 📐 核心公式

```math
\hat{A}_i=\frac{r_i-\text{mean}(r)}{\text{std}(r)+\epsilon}
```

```math
s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{1/|y_i|}
=\exp\!\left(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})}\right)
```

```math
\mathcal{L}_{\text{GSPO}}=-\mathbb{E}\left[\min\!\left(s_i\hat{A}_i,\ \text{clip}(s_i,\,1-\epsilon,\,1+\epsilon)\,\hat{A}_i\right)\right],
\qquad
\epsilon=3\times10^{-4}
```

## 📊 张量流程图

```
# 把重要性比率从 token 级提到序列级
逐 token 对数比 :: [B, G, T] :: log(π_θ / π_old)，每个 token 一个
Σ_t :: 按 mask 求和 → [B, G] :: padding 不参与
/ |y_i| :: 长度归一化 :: 让长短序列可比
s_i = exp(·) :: [B, G] :: 每条序列一个标量比率
目标 :: min(s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i) :: 整条序列只 clip 一次
$ 组内优势 Â_i = (r_i - mean(r)) / (std(r) + ε)，与 GRPO 完全一致
> 奖励是序列级的标量，比率就该是序列级的标量 —— 粒度对齐
> GRPO 的 token 级比率在 MoE 上会放大路由抖动，是训练发散的主因之一
```

## 💻 代码实现

```python
import torch

def sequence_ratio(per_token_logp_new, per_token_logp_old, completion_mask):
    """
    序列级重要性比率（长度归一化）
    per_token_logp_*: [B, G, T]  每个 token 的 log π
    completion_mask:  [B, G, T]  1 = 该 token 属于回答
    返回: [B, G]
    """
    # 1. 逐 token 对数比
    log_ratio = per_token_logp_new - per_token_logp_old        # [B, G, T]

    # 2. 长度归一化后求和，再取 exp
    mask = completion_mask.float()
    length = mask.sum(dim=-1).clamp_min(1.0)                   # [B, G]
    avg_log_ratio = (log_ratio * mask).sum(dim=-1) / length    # [B, G]
    return avg_log_ratio.exp()


def gspo_loss(per_token_logp_new, per_token_logp_old, completion_mask,
              advantages, eps=3e-4):
    """
    GSPO: 整条序列共享一个比率，只 clip 一次
    advantages: [B, G]  组内归一化优势（同 GRPO）
    """
    s = sequence_ratio(per_token_logp_new, per_token_logp_old, completion_mask)
    clipped = torch.clamp(s, 1 - eps, 1 + eps)
    obj = torch.min(s * advantages, clipped * advantages)
    return -obj.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, G, T = 2, 4, 6

    old_logp = torch.randn(B, G, T) - 2.0
    new_logp = old_logp + 0.01 * torch.randn(B, G, T)    # 一次轻微更新

    mask = torch.ones(B, G, T)
    mask[:, :, -2:] = 0                                  # 后 2 个是 padding

    advantages = torch.randn(B, G)
    advantages = (advantages - advantages.mean(-1, keepdim=True)) / \
                 (advantages.std(-1, keepdim=True) + 1e-8)

    s = sequence_ratio(new_logp, old_logp, mask)
    print("序列级比率（应接近 1）:", round(s.mean().item(), 6))
    print("比率范围:", round(s.min().item(), 6), "~", round(s.max().item(), 6))
    print("gspo loss:", gspo_loss(new_logp, old_logp, mask, advantages).item())
```

## 🎯 面试要点

- **一句话**: GSPO = GRPO + 把重要性比率从 token 级换成序列级（长度归一化）
- **为什么 MoE 更需要它**: token 级比率的高方差会放大专家路由抖动，序列级比率把整条序列的更新绑成一个方向，显著更稳
- **长度归一化不可省**: 不归一化的话，长序列的比率会指数级偏离 1，clip 之后梯度全为 0
- **clip 粒度变了，ε 也要跟着变**: 序列级比率偏离 1 的幅度很小，所以 `ε` 取 `3e-4` 量级，比 PPO 的 `0.2` 小几个数量级
- **和 GRPO 的关系**: 目标函数形式完全一样，`min(·, clip(·))` 的骨架没动，只换了比率的定义
- **面试常问**: 序列级和 token 级重要性比率的区别？为什么 MoE 用 token 级比率容易崩？