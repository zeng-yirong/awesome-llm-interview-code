# 交叉熵 / 语言模型损失 (Cross Entropy / LM Loss)

> 下一个 token 预测，LLM 训练基础

## 📌 原理与思想

### 核心概念
语言模型的核心训练目标：给定前文预测下一个 token。通过 shift 操作将 logits 和 labels 对齐，计算交叉熵。交叉熵衡量预测分布与真实分布的差异，梯度计算简洁（softmax + CE 的梯度 = y - one_hot），是 LLM 训练的基础。

### 核心思想
用前面的 token 预测下一个 token。通过 shift 操作：logits 去尾（去掉最后一个位置的预测），labels 去头（去掉第一个位置的目标），使 logits[:, :-1] 预测 labels[:, 1:]。

### 算法步骤
1. Shift logits：shift_logits = logits[:, :-1, :]
2. Shift labels：shift_labels = labels[:, 1:]
3. 展平：flattened_logits = shift_logits.view(-1, V)
4. 展平：flattened_labels = shift_labels.view(-1)
5. 计算交叉熵：loss = CE(flattened_logits, flattened_labels)

## 📐 核心公式

```
L = -Σ log P(xₜ | x<t)

实现: CE(shift(logits), shift(labels))
logits[:, :-1] 预测 labels[:, 1:]

其中:
- Pretrain: 所有 token 参与 loss
- SFT: prompt 部分 label 设为 -100，只算 response
```

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


