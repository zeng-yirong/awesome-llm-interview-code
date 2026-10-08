# LoRA (Low-Rank Adaptation)

> ΔW = BA，低秩分解高效微调

## 📌 原理与思想

冻结预训练权重 W，用低秩矩阵 B·A 近似权重更新 ΔW。A 用高斯初始化，B 用零初始化，保证初始输出不变。

**为什么需要 LoRA？**
- 全量微调大模型需要巨大显存
- LoRA 只训练少量参数（约 0.1%）
- 效果接近全量微调

## 📐 核心公式

```
h = W₀x + ΔWx = W₀x + (B·A)x · (α/r)

W₀: [d, k] 冻结
A:  [r, k] 可训练 (kaiming 初始化)
B:  [d, r] 可训练 (零初始化)
r ≪ min(d, k)

推理时可合并: W_new = W₀ + B·A·(α/r)  → 无额外开销
```

## 📊 张量流程图

```
x: [B, S, k]
  │
  ├── W₀ (frozen) ──────────→ W₀x ──────┐
  │                                       │
  └── dropout → A [r,k] → B [d,r] → BAx · (α/r)
                                              │
                                        ──────┤ +
                                              │
                                        output: [B, S, d]

推理时可合并: W_new = W₀ + B·A·(α/r)  → 无额外开销
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.scaling = alpha / rank
        
        # 冻结原始权重
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.linear.requires_grad_(False)
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)          # B=0 → 初始 ΔW=0
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, S, in_f]"""
        base = self.linear(x)                         # 冻结路径
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora

    def merge(self):
        """推理时合并权重 (无额外开销)"""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

# 在 Transformer 中应用 LoRA
def apply_lora(model, rank=8, alpha=16.0):
    """对模型中的 Q/V 投影应用 LoRA"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'q_proj' in name or 'v_proj' in name:
                # 替换为 LoRA 版本
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha
                )
                lora_layer.linear.weight.data.copy_(module.weight.data)
                # 替换原模块
                parts = name.rsplit('.', 1)
                parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
                setattr(parent, parts[-1], lora_layer)

# 测试
if __name__ == "__main__":
    B, S, D = 2, 10, 512
    x = torch.randn(B, S, D)
    
    lora = LoRALinear(D, D, rank=8, alpha=16.0)
    out = lora(x)
    print(f"Input shape: {x.shape}")   # [2, 10, 512]
    print(f"Output shape: {out.shape}") # [2, 10, 512]
    
    # 验证初始输出等于原始权重输出
    base_out = lora.linear(x)
    diff = (out - base_out).abs().sum()
    print(f"Diff at init (should be 0): {diff.item():.6f}")
```

## 🎯 面试要点

- **B 初始化为零** → 初始时 LoRA 贡献为 0
- **scaling = α/r** 控制 LoRA 更新幅度
- **推理时可 merge**: W = W + BA·scaling → 无额外开销
- **通常只应用于 Q 和 V 的投影矩阵**
- **QLoRA = LoRA + 4bit 量化**，进一步降低显存
- **面试常问**: LoRA 的初始化策略？为什么 B 初始化为零？


