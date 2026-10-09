# LoRA (Low-Rank Adaptation)

> ΔW = BA，低秩分解高效微调

## 📌 原理与思想

冻结预训练权重 W，用低秩矩阵 B·A 近似权重更新 ΔW。A 用高斯初始化，B 用零初始化，保证初始输出不变。

**它解决什么问题**
- 全量微调要存优化器状态：AdamW 下每个参数需要 `m`、`v` 两份 fp32，加上梯度与参数本身，显存约是参数量的 6~8 倍。
- 7B 模型全量微调就要几十 GB，普通显卡根本放不下。
- 但下游适配真的需要改那么多参数吗 —— 权重更新 `ΔW` 的秩往往远低于 `min(d, k)`。

**核心思想**
- 既然 `ΔW` 是低秩的，就不必存整个 `[d, k]` 矩阵，改存两个小矩阵 `B: [d, r]` 与 `A: [r, k]`，`r ≪ min(d, k)`。
- 只训练 `B`、`A`，预训练权重 `W₀` 冻结 —— 可训练参数量降到约 0.1%。
- `B` 用零初始化，保证训练开始时 `BA = 0`，模型初始行为与预训练完全一致，不会一上来就被扰动。

**算法步骤与推导**
- 前向 `h = W₀x + (B·A)x·(α/r)`：主干那条路冻结，分支那条路可训练。
- `A: [r, k]` 用高斯（kaiming）初始化，`B: [d, r]` 用零初始化 —— 一个负责打破对称，一个负责让初始增量为零。
- 缩放因子 `α/r`：调 `r` 时用它把学习率的影响解耦，换 `r` 不必重调 `lr`。
- 推理时可以合并：`W_new = W₀ + B·A·(α/r)`，之后就是一个普通线性层，零额外开销。

**对比与代价**
- 相对全量微调：可训练参数降到约 0.1%，显存与存储都大幅下降，效果接近。
- 代价是表达力受限：秩 `r` 太小就学不下复杂的新任务，`r` 一调大又收益递减而显存回升。
- 能省的是优化器状态和梯度，基线权重 `W₀` 本身还得完整加载 —— 它省的是训练显存，不是推理参数量。

## 📐 核心公式

$$
h=W_0x+\Delta Wx=W_0x+\frac{\alpha}{r}BAx
$$

$$
W_0\in\mathbb{R}^{d\times k}\ \text{(frozen)},\qquad
A\in\mathbb{R}^{r\times k},\qquad
B\in\mathbb{R}^{d\times r},\qquad
r\ll\min(d,k)
$$

$$
A\sim\text{kaiming},\qquad B=\mathbf{0}\ \Rightarrow\ BA=0\ \text{at init}
$$

$$
\frac{\text{trainable}}{\text{total}}=\frac{r(d+k)}{dk}\approx 0.1\%,
\qquad
W_{\text{new}}=W_0+\frac{\alpha}{r}BA\ \ \text{(mergeable at inference)}
$$

## 📊 张量流程图

```
# 主干冻结，旁路低秩可训练
x :: [B, S, k] :: 输入
W₀ :: [d, k] :: 预训练权重，冻结
+ 主干 :: W₀x :: 冻结，不产生梯度
+ 旁路 :: dropout → A[r,k] → B[d,r] → (B·A)x · (α/r) :: 只训练这两个小矩阵
h = W₀x + (B·A)x·(α/r) :: [B, S, d] :: 两条路相加
$ 可训练参数从 d·k 降到 r·(d+k)，而 r ≪ min(d, k)
> B 零初始化 ⇒ 初始 BA = 0，模型一开始与原模型逐位一致
> 推理时可合并：W_new = W₀ + B·A·(α/r)，零额外开销
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


