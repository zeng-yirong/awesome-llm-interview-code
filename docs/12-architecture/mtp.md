# 多 Token 预测 (Multi-Token Prediction)

> 一次预测未来 n 个 token，同一份数据给出 n 倍训练信号

## 📌 原理与思想

### 核心概念
在主模型预测下一个 token 之外，串行接上若干 MTP 模块，每个模块用「上一层 MTP 的隐状态 + 第 `i+k` 个 token 的 embedding」去预测第 `i+1+k` 个 token。训练时提供更密集的监督信号，迫使隐状态包含更长程的前瞻性；推理时这些模块可以直接当投机解码的 draft，几乎白送一个加速器。`Emb` 与 `lm_head` 和主模型共享，参数开销很小；代价是串行结构让模块 `k` 必须等模块 `k-1`，训练时还要多算 `k` 个模块的前向。

### 核心思想
在主干之外串行接上若干 MTP 模块，第 `k` 个模块预测第 `i+1+k` 个 token，让同一份数据提供 `k` 倍的监督信号。模块的输入是「上一层 MTP 的隐状态 + 第 `i+k` 个 token 的 embedding」，逼着隐状态携带更长程的信息。推理时这些模块天然就是一个 draft 模型，可以直接接投机解码 —— 一份结构两处收益。

### 算法步骤
1. 主干输出 `h⁰: [B, T, D]`，接 `lm_head` 预测 `x₂`，这是主损失。
2. 第 `k` 个模块：`h^k = M_k[RMSNorm(h^{k-1}) ; RMSNorm(Emb(x_{i+k}))]`，把上一层隐状态与目标 token 的 embedding 拼起来。
3. 每个模块各接 `lm_head` 预测 `x_{i+1+k}`，得到辅助损失。
4. 总损失 `L = L_main + (λ/D)·Σ_k Σ_i CE(...)`，`λ` 典型取 0.3（前 10T tokens），之后衰减到 0.1。

## 📐 核心公式

$$
h_i^{k}=M_k\!\left[\text{RMSNorm}\!\left(h_i^{k-1}\right);\ \text{RMSNorm}\!\left(\text{Emb}\!\left(x_{i+k}\right)\right)\right],
\qquad
p_{i+k+1}=\text{lm\_head}\!\left(h_i^{k}\right)
$$

$$
\mathcal{L}_{\text{MTP}}=\frac{\lambda}{D}\sum_k\sum_i \text{CE}\!\left(p_{i+k+1},\ x_{i+1+k}\right),
\qquad
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{main}}+\mathcal{L}_{\text{MTP}}
$$

$$
\lambda=0.3\ \text{(first 10T tokens)}\ \longrightarrow\ 0.1,\qquad
\text{Emb},\ \text{lm\_head}\ \text{shared with the main model}
$$

## 📊 张量流程图

```
# 主干之外串行接若干 MTP 模块
x :: [x₁, x₂, x₃, x₄, x₅] :: 输入序列
主干 Transformer :: h⁰ [B, T, D] → lm_head → 预测 x₂ :: 主损失 L_main
+ MTP 模块 1 :: h¹ = M1[h⁰; Emb(x₂)] → 预测 x₃ :: 辅助损失
+ MTP 模块 2 :: h² = M2[h¹; Emb(x₃)] → 预测 x₄ :: 辅助损失
+ MTP 模块 3 :: h³ = M3[h²; Emb(x₄)] → 预测 x₅ :: 辅助损失
$ L = L_main + (λ/D)·Σ_k Σ_i CE(第 k 个预测, 目标)，λ 典型 0.3
> Emb 与 lm_head 与主模型共享，参数开销很小
> 推理时这些模块直接当 draft，几乎白送一个投机解码加速器
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MTPModule(nn.Module):
    """一个 MTP 模块: 拼接上一层隐状态与下一个 token 的 embedding，再过一个 Transformer 块"""

    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)   # 2d → d
        self.block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True)

    def forward(self, h_prev, tok_emb):
        # h_prev:  [B, T, D] 上一层 MTP 的隐状态
        # tok_emb: [B, T, D] 第 i+k 个 token 的 embedding
        x = self.proj(torch.cat([self.norm_h(h_prev), self.norm_e(tok_emb)], dim=-1))
        return self.block(x)                                      # [B, T, D]


def mtp_loss(mtp_modules, h0, input_ids, embed, lm_head, num_heads=1, lam=0.3):
    """
    多 token 预测损失
    h0:        [B, T, D] 主模型最后一层隐状态
    input_ids: [B, T]    输入 token
    返回: (MTP 部分损失, 各层预测的 logits 列表)
    """
    total, h_prev, logits_list = 0.0, h0, []

    for k in range(1, num_heads + 1):
        if input_ids.size(1) - k <= 1:
            break
        target = input_ids[:, 1 + k:]             # 目标: x_{i+1+k}
        tok_emb = embed(input_ids[:, k:-1])       # 输入: x_{i+k}
        h_prev = mtp_modules[k - 1](h_prev[:, :-1], tok_emb)

        logits = lm_head(h_prev)                  # [B, T-k-1, V]
        logits_list.append(logits)
        # 用 sum 再统一除以 D，把 k 个预测的损失归一到同一量级
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum")

    return lam * total / input_ids.size(1), logits_list


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, V = 2, 8, 32, 64

    embed = nn.Embedding(V, D)
    lm_head = nn.Linear(D, V, bias=False)
    mtp_modules = nn.ModuleList([MTPModule(D, 4) for _ in range(2)])

    input_ids = torch.randint(1, V, (B, T))
    h0 = torch.randn(B, T, D)

    loss, logits_list = mtp_loss(mtp_modules, h0, input_ids, embed, lm_head,
                                 num_heads=2, lam=0.3)
    print("MTP 损失:", round(loss.item(), 4))
    for k, lg in enumerate(logits_list, start=1):
        print(f"  第 {k} 层预测形状:", tuple(lg.shape), "← 每深一层少一个位置")

    loss.backward()
    print("梯度已回传:", mtp_modules[0].proj.weight.grad is not None)
```

## 🎯 面试要点

- **和「并行预测头」的区别**: Meta 的早期方案给每个未来位置一个独立头，彼此不依赖；DeepSeek-V3 的 MTP 是**串行**的，第 k 层吃第 k-1 层的隐状态，保留了因果链
- **为什么能提升效果**: 强迫隐状态编码更长程的信息，相当于给模型的「规划能力」加正则
- **参数开销很小**: embedding 和 lm_head 与主模型共享，每个 MTP 模块只有一个投影 + 一个 Transformer 块
- **注意偏移**: 模块的**输入**是 `x_{i+k}`，**目标**是 `x_{i+1+k}` —— 整体比主模型前移 k 步，别把两者搞反
- **为什么要除以 D**: 把 k 个预测的损失归一化回与主损失同一量级，否则 k 越大辅助损失越压过主损失
- **λ 为什么要衰减**: 训练后期主损失收敛，辅助信号的边际价值下降，过大反而干扰
- **推理时的双重身份**: 训练时是辅助损失，推理时是投机解码的 draft —— 一次前向能给多个候选 token
- **面试常问**: MTP 为什么用串行而不是并行头？推理时怎么把它变成加速手段？