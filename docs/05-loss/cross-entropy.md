# 交叉熵 / 语言模型损失 (Cross Entropy / LM Loss)

> 下一个 token 预测，LLM 训练基础

## 📌 原理与思想

语言模型的核心训练目标：给定前文预测下一个 token。通过 shift 操作将 logits 和 labels 对齐，计算交叉熵。

**为什么用交叉熵？**
- 衡量预测分布与真实分布的差异
- 梯度计算简洁：softmax + CE 的梯度 = y - one_hot
- 适合分类任务（词表预测）

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
logits: [B, S, V]  ─→ logits[:, :-1]  → [B, S-1, V]
                                           │
labels: [B, S]       ─→ labels[:, 1:]   → [B, S-1]
                                           │
                                    flatten → CE loss
                                           │
                                     loss: scalar
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


