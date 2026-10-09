# DPO 损失 (Direct Preference Optimization)

> 无需奖励模型，直接优化偏好数据

## 📌 原理与思想

将奖励函数参数化为策略与参考策略的对数比率，直接在偏好数据上优化。增加 chosen 概率，降低 rejected 概率。

**它解决什么问题**
- RLHF 的常规流程要先训一个 reward model，再用 PPO 在线优化，两者都得和策略同规模，显存与调参成本都高。
- PPO 那条流水线很脆：奖励尺度、KL 系数、clip 范围都要调，还容易训崩。
- 而手里拿到的数据往往只是静态偏好对（A 比 B 好），并没有分数 —— 需要一种直接吃偏好对的算法。

**核心思想**
- 带 KL 约束的 RLHF 最优策略有闭式解：`π* ∝ π_ref · exp(r/β)`，反解出来就是 `r = β·log(π*/π_ref) + 常数`。
- 也就是说奖励可以被「策略与参考策略的对数比率」参数化，不必单独训一个 reward model。
- 把这个式子代回偏好损失（Bradley-Terry），常数项自动消掉，最后只剩 chosen 和 rejected 两条回答本身。

**算法步骤与推导**
- 各算两项对数概率：策略与参考策略在 chosen / rejected 上的 `logp`，四项相减得到隐式奖励差 `logits = r_w - r_l`。
- 参考模型的 `logp` 必须 `no_grad`：它只是固定的锚点，不参与更新；只有策略那份需要梯度。
- `-logsigmoid(β·logits)`：chosen 比 rejected 好得越多，sigmoid 越接近 1，loss 越小。
- `β` 控制偏离参考模型的程度，越大越激进，通常取 0.1~0.5。

**对比与代价**
- 相对 PPO：省掉 reward model 和在线 rollout，一次前向就能算出 loss，工程复杂度大幅下降。
- 代价是完全 off-policy：只能吃固定的偏好数据集，无法在线探索；数据分布一旦偏离当前策略，提升就受限。
- 它还需要一份额外的参考模型副本常驻显存，这一点和 PPO 一样躲不掉。

## 📐 核心公式

$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\!\left(\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right)\right]
$$

$$
\text{logits}=\left(\log p_\theta^{w}-\log p_{\text{ref}}^{w}\right)-\left(\log p_\theta^{l}-\log p_{\text{ref}}^{l}\right),
\qquad
\mathcal{L}=-\log\sigma\!\left(\beta\cdot\text{logits}\right)
$$

$$
y_w:\ \text{chosen},\qquad y_l:\ \text{rejected},\qquad \beta\in[0.1,\ 0.5]
$$

## 📊 张量流程图

```
# 隐式奖励：策略相对参考策略的对数比率
+ chosen :: policy_logp_w - ref_logp_w = r_w :: 优选回答的隐式奖励
+ rejected :: policy_logp_l - ref_logp_l = r_l :: 拒绝回答的隐式奖励
logits = r_w - r_l :: chosen 比 rejected 好多少
loss = -logsigmoid(β · logits) :: logits 越大 loss 越小
$ r = β·log(π/π_ref) + 常数 —— 把 RLHF 的闭式解代回 Bradley-Terry
> 参考模型的 logp 必须 no_grad，它只是固定锚点
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO 损失
    
    *_logps: [batch_size] 每个样本的对数概率之和
    """
    # 隐式奖励 = log(π_θ / π_ref)
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    # 奖励差
    logits = chosen_ratio - rejected_ratio

    # DPO 损失
    return -F.logsigmoid(beta * logits).mean()

# 完整训练流程
def compute_logps(model, input_ids, labels):
    """计算模型对 labels 的对数概率之和"""
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    labels = labels[:, 1:]
    per_token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps.sum(dim=-1)

# 测试
if __name__ == "__main__":
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)
    
    loss = dpo_loss(policy_chosen, policy_rejected,
                    ref_chosen, ref_rejected, beta=0.1)
    print(f"DPO Loss: {loss.item():.4f}")
```

## 🎯 面试要点

- **核心**: 奖励 = log(π_θ / π_ref)，无需显式奖励模型
- **chosen 概率↑ + rejected 概率↓** → 对齐人类偏好
- **β 控制偏离参考模型程度**，通常 0.1-0.5
- **比 PPO 简单很多**: 不需要 reward model 和 critic
- **变体**: IPO, KTO, ORPO, SimPO 等
- **面试常问**: DPO 和 PPO 的区别？为什么 DPO 更简单？


