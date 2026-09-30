# 混合精度训练 (Mixed Precision Training)

> FP16/BF16 计算 + FP32 主权重

## 📌 原理与思想

使用前向/反向用 FP16/BF16 减少显存和加速计算，但保持 FP32 主权重防止精度损失。配合 loss scaling 防止梯度下溢。

**为什么需要混合精度？**
- FP16/BF16: 2 bytes，显存减半，计算加速
- FP32: 4 bytes，精度高
- 混合使用：既快又准

## 📐 核心公式

```
前向/反向: FP16 (半精度, 2 bytes)
主权重: FP32 (全精度, 4 bytes)
梯度: FP16 → loss_scale → 转 FP32 → 更新主权重

BF16: 指数位更多 (8 vs 5), 不需要 loss scaling

精度对比:
  FP32: 1 + 8 + 23 bits (符号 + 指数 + 尾数)
  FP16: 1 + 5 + 10 bits
  BF16: 1 + 8 + 7 bits
```

## 📊 张量流程图

```
FP32 主权重 ──→ copy ──→ FP16 权重 ──→ 前向 ──→ FP16 loss
                                    ↑                      │
                                    │               loss_scale × loss
                                    │                      │
FP32 主权重 ←── update ←── FP32 梯度 ←── cast ←── FP16 梯度 ←─┘
```

## 💻 代码实现

```python
import torch

# 方法1: PyTorch 原生 AMP (Automatic Mixed Precision)
scaler = torch.amp.GradScaler('cuda')

for batch in dataloader:
    optimizer.zero_grad()
    
    # 自动将操作转为 FP16
    with torch.amp.autocast('cuda'):
        loss = model(batch)
    
    # Loss scaling 防止梯度下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 方法2: BF16 更简单 (不需要 loss scaling)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()   # 直接反向，不需要 scaler
optimizer.step()

# 方法3: HuggingFace Trainer 自动处理
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir='./output',
    fp16=True,           # 使用 FP16
    # bf16=True,         # 或使用 BF16 (A100/H100 推荐)
    fp16_full_eval=True, # 评估也用 FP16
)
trainer = Trainer(model=model, args=args)
trainer.train()

# 验证混合精度
def check_mixed_precision(model):
    """检查模型是否使用混合精度"""
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
        # 应该看到大部分参数是 fp16/bf16
        # 但某些关键参数（如 LayerNorm）保持 fp32
```

## 🎯 面试要点

- **FP16**: 1+5+10 bits, 需要 loss scaling 防梯度下溢
- **BF16**: 1+8+7 bits, 指数位多 → 不需要 loss scaling
- **显存减半 + 计算加速** (Tensor Core)
- **现代 GPU (A100/H100) 推荐 BF16**
- **主权重始终 FP32** 保证精度
- **面试常问**: FP16 和 BF16 的区别？为什么 BF16 不需要 loss scaling？

---

**来源**: [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)
