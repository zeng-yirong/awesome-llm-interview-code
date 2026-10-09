# DPO 损失 (Direct Preference Optimization)

> 无需奖励模型，直接优化偏好数据

## 📌 原理与思想

### 核心概念
将奖励函数参数化为策略与参考策略的对数比率，直接在偏好数据上优化。增加 chosen 概率，降低 rejected 概率。相比 PPO 不需要 reward model 和 critic，训练更简单稳定，效果相当甚至更好。

### 核心思想
利用 RL 的对偶性，将奖励函数隐式表示为 r(x,y) = β · log(πθ(y|x)/πref(y|x))。这样策略优化目标可以直接用策略的对数概率差来表示，无需显式奖励模型。

### 算法步骤
1. 收集偏好数据：(prompt, chosen_response, rejected_response)
2. 计算策略与参考策略的对数概率差
3. 构造损失函数：增加 chosen 概率，降低 rejected 概率
4. 直接用梯度下降优化策略模型

## 📐 核心公式

```
L_DPO = -E[log σ(β · (log πθ(yw|x)/πref(yw|x) - log πθ(yl|x)/πref(yl|x)))]

简化:
logits = (logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected)
loss = -logsigmoid(β · logits)

其中:
- yw: chosen（优选）回答
- yl: rejected（拒绝）回答
- β: 控制偏离参考模型的程度，通常 0.1-0.5
```

## 📊 张量流程图

```
chosen:     policy_logp_w - ref_logp_w  = r_w  (隐式奖励)
rejected:   policy_logp_l - ref_logp_l  = r_l

logits = r_w - r_l     (chosen 比 rejected 好多少)
loss = -logsigmoid(β × logits)     (越大越好 → loss 越小)

β 控制偏离参考模型的程度
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


