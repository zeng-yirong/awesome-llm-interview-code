# 梯度检查点 (Gradient Checkpointing)

> 时间换空间，重新计算代替存储激活

## 📌 原理与思想

不保存所有中间激活值，只保存检查点。反向传播时重新计算需要的激活值。用约 20% 额外计算换取大量显存。

**它解决什么问题**
- 反向传播需要前向的中间激活，标准做法是把每一层的激活全存下来，显存随层数按 `O(L)` 线性增长。
- 层数一多，激活本身就变成训练显存的大头，甚至超过参数与优化器状态。
- 而算力和显存是一对矛盾：显存不够时只能减 batch、减序列长度，直接拖慢吞吐。

**核心思想**
- 少存一点，用时再算一遍：只在若干位置留「检查点」，中间那些激活反向时当场重算。
- 检查点取 `√L` 份时总开销最优：存 `√L` 份激活，重算代价也是 `√L` 量级，显存降到 `O(√L)`。
- 重算的只是前向的一部分，多出来的计算量远小于把 batch 砍掉带来的损失。

**算法步骤与推导**
- 把网络切成若干段，每段边界保存一份激活作为检查点，段内中间的激活一律丢掉。
- 前向只算到检查点边界，段内激活用完即弃。
- 反向走到某段时，从该段的检查点重新跑一次前向，把需要的激活重建出来，再正常回传。
- 段内那一段等于多跑了一遍前向，整体训练时间增加约 20%，而激活显存从 `O(L)` 降到 `O(√L)`。

**对比与代价**
- 相对标准训练：激活显存大幅下降，省下来的显存可以直接换成更大的 batch 或更长的序列。
- 代价是训练时间增加约 20%，典型的「拿时间换显存」。
- 段分得越粗省得越多但重算越贵，`√L` 是这条曲线上的最优点。

## 📐 核心公式

$$
\text{activations}: O(L)\ \longrightarrow\ O\!\left(\sqrt{L}\right)
$$

$$
\text{memory}: L\,B\,S\,d\cdot 4\ \text{B}\ \longrightarrow\ \sqrt{L}\,B\,S\,d\cdot 4\ \text{B}
$$

$$
\text{checkpoints}=\sqrt{L},\qquad \text{time overhead}\approx 20\%
$$

## 📊 张量流程图

```
# 标准训练：全部激活都留着
+ 标准 :: Layer₁ → act₁ → Layer₂ → act₂ → … → Layer_L → act_L :: 每层激活全部保存
! ✗ 标准代价：激活显存 O(L)，层数一多就成了显存大头
! ✓ 检查点：只保存 √L 份，段内激活反向时重算
Layer₁ → act₁* → Layer₂ → Layer₃ → act₃* → … :: 带 * 的是检查点
! 反向时从最近的检查点重跑一次前向，把段内激活重建出来
$ 激活显存 O(L) → O(√L)，代价是训练时间 +20%
> 省下来的显存可以直接换成更大的 batch 或更长的序列
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


