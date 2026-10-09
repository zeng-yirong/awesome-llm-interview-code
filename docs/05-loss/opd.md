# 在线策略蒸馏 (On-Policy Distillation)

> 学生自己生成轨迹，教师逐 token 给稠密监督

## 📌 原理与思想

让学生模型自己采样生成轨迹，再让教师模型在学生实际走过的每个 token 上给出完整分布作为监督信号。相比用教师生成的静态数据做离线蒸馏，on-policy 训练消除了训练与推理的分布不匹配（exposure bias）。而相比只有稀疏结果奖励的 RL，教师的逐 token 分布本身就是一个稠密奖励，不需要额外的 reward model。

**它解决什么问题**
- 离线蒸馏（SFT on teacher data）里，学生训练时见的是教师的完美轨迹，推理时却要基于自己犯过的错继续往下生成，误差沿序列累积。
- 只有稀疏结果奖励的 RL 里，整条序列只有一个标量奖励，token 级信用分配困难，还得额外训一个 reward model。
- 两条路都缺同一样东西：一个既落在学生自己的分布上、又足够稠密的监督信号。

**核心思想**
- 让学生自己采样，教师在被采样出来的轨迹上逐 token 给出完整分布 —— 监督落在学生真正会走的路径上。
- 教师的分布在每个位置都是一个几万维的概率向量，本身就是一个稠密奖励，不需要 reward model。
- 训练分布和推理分布从此一致，exposure bias 被从根上消掉。

**算法步骤与推导**
- 学生前向并采样得到 `y_S`，它带着学生自己的错误 —— 这正是要让它学会纠正的地方。
- 教师在同一批 `prompt + y_S` 上做一次前向，得到 `teacher_logits: [B, T, V]`，必须 `detach` / `no_grad`。
- 逐 token 算散度 `D(p_T ‖ p_S)`，得到 `per_token_loss: [B, T]`，再按有效 token 数求平均。
- 用广义 JSD 插值统一方向：`β = 0` 是前向 KL（mode-covering）、`β = 1` 是反向 KL（mode-seeking）、`β = 0.5` 就是标准 JSD。

**对比与代价**
- 相对离线蒸馏：监督落在学生自己的分布上，暴露偏差不再累积；代价是每个 batch 都要现场采样，不能预先把数据处理好。
- 相对稀疏奖励 RL：不用训 reward model，token 级信用分配天然解决；代价是必须有一个更强的教师模型可用。
- 反向 KL 是 mode-seeking，学生只学教师的高概率模式，不覆盖教师的全部尾巴 —— 方向与 Hinton 蒸馏正好相反。

## 📐 核心公式

```
损失: L_OPD = E_{y ~ π_S(·|x)} [ (1/|y|) Σ_t D( p_T(·|x, y_<t) ‖ p_S(·|x, y_<t) ) ]

广义 JSD 插值 (β ∈ [0, 1]):
  M = β·p_T + (1 - β)·p_S
  D_GJS(β) = (1 - β)·KL(p_T ‖ M) + β·KL(p_S ‖ M)

端点 (可用代码断言验证):
  β = 0   → KL(p_T ‖ p_S)   前向 KL (mode-covering，Hinton 蒸馏的方向)
  β = 1   → KL(p_S ‖ p_T)   反向 KL (mode-seeking，只学教师的高概率模式)
  β = 0.5 → 标准 JSD
```

其中:
- `y`: **学生自己采样**出来的序列（不是教师生成的，这是 on-policy 的关键）
- `p_T` / `p_S`: 教师 / 学生在位置 t 的 next-token 分布，均在词表 V 上
- `D`: 逐 token 的散度，在词表维度上求和
- `β`: 控制 KL 方向；实践里常用反向 KL 一侧，它只要求学生匹配教师的高概率模式

## 📊 张量流程图

```
# 两条路的对比
! ✗ 离线蒸馏：教师生成 y_T，学生拟合 y_T，训练分布 ≠ 推理分布
! ✓ on-policy 蒸馏：学生自己采样，教师在自己的轨迹上逐 token 打分
# 学生采样：监督落在学生真正会走的路径上
prompt x :: 同一批输入
y_S ~ π_S(·|x) :: [B, T] :: 学生采样得到，带着自己的错误
# 教师打分：逐 token 给出完整分布
teacher_logits = π_T(·|x, y_<t) :: [B, T, V] :: 必须 detach / no_grad
student_logits = π_S(·|x, y_<t) :: [B, T, V] :: 需要梯度
per_token_loss = D(p_T ‖ p_S) :: [B, T] :: 逐 token 散度
loss = mean(per_token_loss) :: 按有效 token 数平均，得到标量
$ β=0 → 前向 KL（mode-covering）、β=1 → 反向 KL（mode-seeking）、β=0.5 → 标准 JSD
> 教师的逐 token 分布本身就是稠密奖励，不需要 reward model
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F

def generalized_jsd(student_logits, teacher_logits, beta=1.0, temperature=1.0):
    """
    广义 JSD 插值
    beta = 0   → 前向 KL  KL(p_T ‖ p_S)   (mode-covering)
    beta = 1   → 反向 KL  KL(p_S ‖ p_T)   (mode-seeking)
    beta = 0.5 → 标准 JSD
    student_logits, teacher_logits: [B, T, V]，未归一化的 logits
    返回逐 token 散度 [B, T]
    """
    # 1. 温度缩放 + log_softmax（数值稳定）
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_p = s_logp.exp()
    t_p = t_logp.exp()

    # 2. 混合分布 M = beta * p_T + (1 - beta) * p_S
    #    clamp_min 只是防止 log(0)，V 个 token 上概率不会真的到 0
    m_logp = (beta * t_p + (1 - beta) * s_p).clamp_min(1e-8).log()

    # 3. KL(p_T ‖ M) 与 KL(p_S ‖ M)，在词表维度求和
    kl_t = (t_p * (t_logp - m_logp)).sum(dim=-1)     # [B, T]
    kl_s = (s_p * (s_logp - m_logp)).sum(dim=-1)     # [B, T]

    return (1 - beta) * kl_t + beta * kl_s           # [B, T]


def opd_loss(student_logits, teacher_logits, loss_mask=None,
             beta=1.0, temperature=1.0, top_k=0):
    """
    在线策略蒸馏损失
    loss_mask: [B, T]，1 = 学生在该位置真实采样出的 token
    top_k:     >0 时只保留教师概率最高的 k 个 token
               （黑盒教师只吐 top-k logits 时用，其余位置截断）
    """
    # 教师永远不参与反传，否则「教师」会被同一步更新带跑
    teacher_logits = teacher_logits.detach()

    if top_k > 0:
        # 教师只给 top-k: 其余位置截断到一个很大的负数
        # 用 -1e9 而不是 -inf，避免 0 * inf 产生 nan
        kth = teacher_logits.topk(top_k, dim=-1).values[..., -1:]
        teacher_logits = teacher_logits.masked_fill(teacher_logits < kth, -1e9)

    per_token = generalized_jsd(student_logits, teacher_logits, beta, temperature)

    if loss_mask is None:
        return per_token.mean()
    mask = loss_mask.float()
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 2, 4, 8
    student = torch.randn(B, T, V)
    teacher = torch.randn(B, T, V)

    # 端点自检: beta=0 应等于前向 KL, beta=1 应等于反向 KL
    s_logp = F.log_softmax(student, dim=-1)
    t_logp = F.log_softmax(teacher, dim=-1)
    # F.kl_div(input, target, log_target=True) = KL(target ‖ input)
    fwd = F.kl_div(s_logp, t_logp, log_target=True, reduction='none').sum(-1)
    rev = F.kl_div(t_logp, s_logp, log_target=True, reduction='none').sum(-1)

    print("beta=0  :", generalized_jsd(student, teacher, beta=0.0).mean().item())
    print("  前向KL:", fwd.mean().item())
    print("beta=1  :", generalized_jsd(student, teacher, beta=1.0).mean().item())
    print("  反向KL:", rev.mean().item())
    print("beta=0.5:", generalized_jsd(student, teacher, beta=0.5).mean().item(), "(标准 JSD)")

    # top-k 教师 + mask
    mask = torch.ones(B, T)
    mask[:, -1] = 0
    print("top_k=3:", opd_loss(student, teacher, mask, beta=1.0, top_k=3).item())
```

## 🎯 面试要点

- **为什么要 on-policy**: 消除 exposure bias。离线蒸馏的误差累积是 O(T²)，on-policy 降到 O(T)
- **为什么常选反向 KL**: 反向 KL 是 mode-seeking，学生只去匹配教师的高概率模式；前向 KL 是 mode-covering，会给教师几乎不给概率的区域也分配质量
- **KL 方向决定行为**: 前向 KL 覆盖所有模式但分布更散（易产生幻觉区域），反向 KL 更尖锐但可能丢多样性
- **与 RL 的区别**: 不需要 reward model，教师逐 token 的分布就是稠密奖励；同等效果下比 RL 便宜得多
- **教师的 logits 必须 detach**: 教师是「固定靶」，只有学生一侧回传梯度
- **黑盒教师**: 拿不到全词表 logits 时，用 top-k logits 截断近似，其余位置填一个很大的负数（不要填 `-inf`，会在 `0 * inf` 处产生 nan）
- **广义 JSD 一个公式统一了前后向 KL**: `β` 从 0 滑到 1，就是从 mode-covering 滑到 mode-seeking，代码里用端点数值断言自检
- **面试常问**: on-policy 蒸馏和 RLHF 是什么关系？为什么 OPD 不需要 reward model？