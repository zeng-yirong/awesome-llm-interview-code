# 在线自蒸馏 (On-Policy Self-Distillation)

> 同一个模型既是教师又是学生，用特权上下文造出更强的自己

## 📌 原理与思想

不再依赖外部更强的教师模型：让学生自己采样，同时把同一份权重在特权上下文（参考答案、关键提示、解题方向等）条件下的分布当作教师分布。学生在没有特权信息的条件下学习，把训练时才有的额外信息转成训练信号。教师与学生共享参数，因此既不需要额外的教师显存，也不存在师生能力差距过大导致的负迁移。

**它解决什么问题**
- OPD 需要外部更强的教师，可到了 SOTA 之上往往已经没有更强的模型可用。
- 特权信息（参考答案、工具返回、用户纠正）推理时拿不到、训练时拿得到，白放着浪费。
- 找外部教师还会引入师生能力差距：差距太大时学生学不动，反而出现负迁移。

**核心思想**
- 把「更强的模型」换成「信息更全的自己」：同一份权重 `θ`，一边看 prompt，一边额外拼接特权上下文 `c`。
- 两条路径共享参数，教师那条只是多了一段输入，因此既不需要额外显存，也不存在能力差距。
- 学生学的是「在没有特权信息的条件下逼近有特权信息时的分布」，等于把训练期才有的信息蒸馏进参数。

**算法步骤与推导**
- 学生路径 `π_θ(·|x)` 必须真采样得到 `y ~ π_θ`，这条路径需要梯度。
- 教师路径 `π_θ(·|x ⊕ c)` 用同一份权重，只需一次前向，`no_grad` 即可。
- 逐 token 反向 KL：`KL(π_θ(·|x, y_<t) ‖ π_θ(·|x ⊕ c, y_<t))`，只在 `y` 自己的 token 上回传。
- 用反向 KL（mode-seeking）而不是前向：学生只对齐「信息更全的自己」的高概率模式，不强行覆盖全部尾巴。

**对比与代价**
- 相对 OPD：省掉外部教师模型的显存与部署，也不再有师生能力差距导致的负迁移。
- 特权上下文的构造成了新的依赖：参考答案、提示、工具返回都得由数据管线稳定提供，质量直接决定上限。
- 教师与学生同源，能提供的额外信息上限就是特权上下文本身 —— 没有外部知识注入时收益明显变小。

## 📐 核心公式

$$
\text{student}: \pi_\theta(\cdot\mid x)\ \ \text{(rollout, gradient)},
\qquad
\text{teacher}: \pi_\theta(\cdot\mid x\oplus c)\ \ \text{(one forward, no\_grad)}
$$

$$
\mathcal{L}=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\left[\frac{1}{|y|}\sum_t\operatorname{KL}\!\left(\pi_\theta(\cdot\mid x,y_{<t})\ \|\ \pi_\theta(\cdot\mid x\oplus c,y_{<t})\right)\right]
$$

$$
\text{forward KL}: \operatorname{KL}(\pi_T\|\pi_S)
\qquad\text{vs}\qquad
\text{reverse KL}: \operatorname{KL}(\pi_S\|\pi_T)
$$

$$
c\in\{\text{reference answer},\ \text{hint},\ \text{strategy name},\ \text{tool output},\ \text{user correction}\}
$$

## 📊 张量流程图

```
# 同一份权重 θ，两条路径
学生路径 :: x → π_θ(·|x) :: 需要真采样 rollout，带着自己的错误
教师路径 :: x ⊕ c → π_θ(·|x ⊕ c) :: 同一份权重多拼一段上下文，no_grad
c :: 参考答案 / 关键提示 / 解题策略 / 工具返回 / 用户纠正
逐 token 反向 KL :: KL(π_θ(·|x, y_<t) ‖ π_θ(·|x ⊕ c, y_<t)) :: 只在 y 自己的 token 上回传
$ 教师不是「更强的模型」，而是「信息更全的同一个自己」
> 师生共享参数：不需要额外的教师显存，也不存在能力差距导致的负迁移
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F
from collections import namedtuple

LMOutput = namedtuple("LMOutput", ["logits"])


@torch.no_grad()
def teacher_forward(model, input_ids, privileged_ids):
    """教师 = 同一模型 + 特权上下文，只做一次前向，不需要梯度"""
    full = torch.cat([privileged_ids, input_ids], dim=1)
    logits = model(full).logits
    # 只取答案段（去掉特权上下文那一段）
    return logits[:, privileged_ids.size(1):]


def self_opd_loss(model, input_ids, privileged_ids, answer_mask):
    """
    在线自蒸馏损失
    input_ids:      [B, T] 学生看到的 prompt + 学生自己采样的回答
    privileged_ids: [B, P] 特权上下文（如参考答案），只拼给教师
    answer_mask:    [B, T] 1 = 学生在该位置真实生成的 token
    """
    # 1. 学生前向（需要梯度）—— 与 rollout 时是同一批 token
    student_logits = model(input_ids).logits                      # [B, T, V]

    # 2. 教师前向：同一权重 + 特权上下文，no_grad
    t_logits = teacher_forward(model, input_ids, privileged_ids)   # [B, T, V]

    # 3. 逐 token 反向 KL: KL(p_S ‖ p_T)
    s_logp = F.log_softmax(student_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)            # [B, T]

    # 4. 只在学生自己生成的 token 上回传
    mask = answer_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)


if __name__ == "__main__":
    class ToyLM(torch.nn.Module):
        """玩具模型: 真实场景是 HuggingFace 的 CausalLM，这里只验证流程"""
        def __init__(self, vocab=16, dim=8):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, vocab)

        def forward(self, ids):
            return LMOutput(self.head(self.emb(ids)))   # [B, T, V]

    torch.manual_seed(0)
    model = ToyLM()
    input_ids = torch.randint(0, 16, (2, 6))
    privileged = torch.randint(0, 16, (2, 3))

    answer_mask = torch.zeros(2, 6)
    answer_mask[:, 3:] = 1.0          # 后 3 个 token 是学生自己生成的

    loss = self_opd_loss(model, input_ids, privileged, answer_mask)
    print("self-opd loss:", loss.item())

    loss.backward()
    print("学生一侧收到梯度:", model.head.weight.grad is not None)
    print("输出形状:", (2, 6), "-> 标量")
```

## 🎯 面试要点

- **与 OPD 的核心区别**: OPD 的教师是另一个更强的模型；self-OPD 的教师是同一份权重 + 特权上下文
- **为什么需要它**: OPD 需要外部更强的教师，但到了 SOTA 之上往往没有更强的模型可用；而特权信息训练时拿得到、推理时拿不到，正好当监督信号
- **特权上下文的粒度很关键**: 给完整参考答案不一定最优 —— 中间抽象（解题策略名、方法方向、问题类别）往往在更少 hint token 下效果更好，因为完整答案会让教师分布过分偏离学生当前能力
- **必须是同一个模型**: 教师和学生共享参数，所以不存在「师生能力差距过大」导致的负迁移
- **rollout 必须 on-policy**: 训练 token 要来自学生自己的采样，否则退化成普通的上下文蒸馏
- **已知副作用**: 反向 KL 是 mode-seeking，长期训练会压缩输出多样性、让模型变「刚性」，需要靠 hint 设计、散度方向和训练步数来调节
- **用武之地**: 持续学习（把推理时的信息固化成权重）、利用隐式用户反馈做定向纠错
- **面试常问**: 没有外部教师时怎么蒸馏？特权信息在推理时拿不到，为什么还能当监督信号？