# 交叉熵 / 语言模型损失 (Cross Entropy / LM Loss)

> 下一个 token 预测，LLM 训练基础

## 📌 原理与思想

语言模型的核心训练目标：给定前文预测下一个 token。通过 shift 操作将 logits 和 labels 对齐，计算交叉熵。

**它解决什么问题**
- 语言模型要学的只有一件事：给定前文，下一个 token 应该是什么。这件事必须落成一个可求导、可批量计算的标量损失。
- 直接最大化整句的联合概率要连乘几十个小于 1 的概率，数值会下溢；取对数变成连加，才是能算的形式。
- 词表有几万维，预测本质上是一个超大分类问题，需要一个天然适配分类的损失函数。

**核心思想**
- 把「下一个 token 的概率」转成「正确 token 的负对数概率」：概率越接近 1，loss 越接近 0。
- 每个位置上只有一个正确答案，所以交叉熵退化成取正确 token 那一项的负对数，不必真的构造几万维的 one-hot 向量。
- 对每个位置都算一遍，整条序列的 loss 就是这些位置的平均，也就把「预测下一个 token」变成了可微的目标。

**算法步骤与推导**
- `shift` 对齐：`logits[:, :-1]` 配 `labels[:, 1:]`，位置 `t` 的输出预测的是位置 `t+1` 的 token。少这一步就等于把答案直接喂给模型。
- 对 logits 做 `log_softmax` 直接得到对数概率，避免「先 `softmax` 再 `log`」在概率极小时下溢成 `-inf`。
- 取出正确 token 位置的对数值、取负、按有效 token 求平均。SFT 时把 prompt 段的 label 置为 `-100`，`ignore_index` 会把它们排除出分母。
- 梯度形式极简：`softmax - one_hot`。预测得越离谱梯度越大，且自带归一化，不必额外调损失尺度。

**对比与代价**
- 对比 MSE 这类回归损失：交叉熵的梯度不会在概率饱和后消失，即使 `p` 已接近 0 仍有量级可观的拉力。
- 代价是它只看正确 token 那一项，对剩下几万维的分布毫无约束 —— 想让模型学会完整的分布（蒸馏）就得换成 KL 之类的度量。
- 按序列平均会稀释长回答里每个 token 的梯度，DAPO 改成 token 级求和正是为了修掉这一点。

## 📐 核心公式

$$
\mathcal{L}_{\text{CE}}=-\frac{1}{|x|}\sum_{t=1}^{|x|}\log P_\theta(x_t\mid x_{<t})
$$

$$
\text{CE}\!\left(\operatorname{shift}(\text{logits}),\ \operatorname{shift}(\text{labels})\right),
\qquad
\text{logits}[:,:-1]\ \text{predicts}\ \text{labels}[:,1:]
$$

$$
\begin{aligned}
\text{Pretrain}: &\quad \text{all tokens counted}\\[2pt]
\text{SFT}: &\quad \text{label}_{\text{prompt}}=-100,\qquad \text{ignore\_index}=-100
\end{aligned}
$$

## 📊 张量流程图

```
# shift 对齐：位置 t 的输出预测位置 t+1 的 token
logits :: [B, S, V] :: 每个位置对整个词表的打分
labels :: [B, S] :: 真实 token 下标，prompt 段置 -100
+ 预测 :: logits[:, :-1] → [B, S-1, V] :: 丢掉最后一个位置，它没有下一个 token
+ 目标 :: labels[:, 1:] → [B, S-1] :: 丢掉第一个位置，它没有被谁预测
logp = log_softmax(logits, -1) :: [B, S-1, V] :: 直接取对数概率，避免下溢
loss = -mean(logp[range, 目标]) :: 只取正确 token 那一项，结果是标量
$ 梯度 = softmax(logits) - one_hot(目标)，预测越离谱梯度越大
> SFT 时 prompt 段的 label 置 -100，ignore_index 把它们排除出分母
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F

def lm_loss(logits, labels, ignore_index=-100):
    """
    语言模型损失（下一个 token 预测）
    
    logits: [B, S, vocab_size]
    labels: [B, S]
    """
    # Shift: 用前面的 token 预测下一个
    shift_logits = logits[:, :-1, :].contiguous()   # [B, S-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, S-1]
    
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )

# 测试
if __name__ == "__main__":
    B, S, V = 2, 10, 32000
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    
    loss = lm_loss(logits, labels)
    print(f"Loss: {loss.item():.4f}")

# SFT Loss: 只计算 response 部分
def sft_loss(logits, labels, prompt_lengths):
    """
    SFT 损失：prompt 部分设为 -100
    """
    masked_labels = labels.clone()
    for i, plen in enumerate(prompt_lengths):
        masked_labels[i, :plen] = -100
    return lm_loss(logits, masked_labels)
```

## 🎯 面试要点

- **Shift 操作是核心**: logits 去尾，labels 去头
- **ignore_index=-100** 的位置不参与梯度
- **Pretrain**: 所有 token 参与 loss
- **SFT**: prompt 部分 label 设为 -100，只算 response
- **面试常问**: Pretrain Loss 和 SFT Loss 的区别？


