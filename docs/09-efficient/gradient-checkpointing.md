# 梯度检查点 (Gradient Checkpointing)

> 时间换空间，重新计算代替存储激活

## 📌 原理与思想

不保存所有中间激活值，只保存检查点。反向传播时重新计算需要的激活值。用约 20% 额外计算换取大量显存。

**为什么需要梯度检查点？**
- 标准训练: 保存所有层激活 → 显存 O(L)
- 检查点: 只保存 √L 个检查点 → 显存 O(√L)
- 代价: 反向传播需要重新计算 → 训练时间 +20%

## 📐 核心公式

```
标准: 保存所有层激活 → 显存 O(L)
Checkpoint: 只保存 √L 个检查点 → 显存 O(√L)

代价: 反向传播需要重新计算 → 训练时间 +20%

显存节省:
  标准: L × batch_size × seq_len × d_model × 4 bytes
  检查点: √L × batch_size × seq_len × d_model × 4 bytes
```

## 📊 张量流程图

```
标准训练:
  Layer₁ → act₁ → Layer₂ → act₂ → ... → Layer_L → act_L
  全部保存 → 显存 O(L)

梯度检查点:
  Layer₁ → act₁* → Layer₂ → Layer₃ → act₃* → ...
              ↑ 检查点 (保存)              ↑ 检查点
  反向时: 从检查点重新计算 → 显存 O(√L)
  代价: 多一次前向计算 → 训练慢 ~20%
```

## 💻 代码实现

```python
import torch
from torch.utils.checkpoint import checkpoint

# 方法1: 使用 torch.utils.checkpoint
def custom_forward(x, weight1, weight2):
    return torch.relu(x @ weight1) @ weight2

# 不保存中间激活，反向时重新计算
output = checkpoint(custom_forward, x, w1, w2, use_reentrant=False)

# 方法2: HuggingFace 模型直接开启
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "model_name",
    gradient_checkpointing=True    # 一行开启
)

# 方法3: 手动实现
class CheckpointBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x, use_reentrant=False)

# 在 Transformer 中使用
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x, mask=None):
        # 使用 checkpoint 包裹
        x = x + checkpoint(self.attn, self.norm1(x), mask, use_reentrant=False)
        x = x + checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        return x

# 测试显存节省
def measure_memory(use_checkpoint=False):
    """测量显存使用"""
    import torch.cuda
    
    model = build_model(use_checkpoint=use_checkpoint)
    x = torch.randn(4, 2048, 4096, device='cuda')
    
    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory: {peak_memory:.2f} GB")
    return peak_memory

# 对比
print("Without checkpoint:")
mem1 = measure_memory(use_checkpoint=False)

print("\nWith checkpoint:")
mem2 = measure_memory(use_checkpoint=True)

print(f"\nMemory saved: {(mem1 - mem2):.2f} GB ({(1 - mem2/mem1)*100:.1f}%)")
```

## 🎯 面试要点

- **用 ~20% 额外计算时间换取大量显存**
- **显存从 O(L) 降到 O(√L)**
- **HuggingFace 模型一行开启**: `gradient_checkpointing=True`
- **常与 LoRA 配合**: LoRA 减少参数 + 检查点减少激活
- **use_reentrant=False** 是新版推荐设置
- **面试常问**: 梯度检查点的原理？代价是什么？


