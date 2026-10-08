# 混合专家模型 (Mixture of Experts, MoE)

> 稀疏激活，大参数量小计算量

## 📌 原理与思想

将 FFN 替换为多个"专家"网络，Router 为每个 token 选择 Top-K 个专家处理。总参数量大但每次只激活部分，计算量可控。

**为什么需要 MoE？**
- 增加模型容量（参数量）但不增加计算量
- 每个 token 只激活 K 个专家
- Mixtral 8x7B: 46.7B 参数，实际计算量约 12.9B

## 📐 核心公式

```
MoE(x) = Σᵢ∈TopK softmax(Router(x))ᵢ · Eᵢ(x)

Mixtral 8x7B: 8 个专家选 2 个
实际计算量 ≈ 12.9B (vs 46.7B 总参数)
```

## 📊 张量流程图

```
x: [B*S, D]
  │
  ├── Router: [D, N_experts] → logits: [B*S, N]
  │                                    │
  │                              Top-K → indices, weights
  │                                    │
  └── Experts: E₁, E₂, ..., Eₙ        │
        │                              │
   每个 expert 处理分配到的 token       │
        │                              │
   weighted_sum(expert_out × weight) ←─┘
        │
   output: [B*S, D] → [B, S, D]
```

## 💻 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4*d_model), nn.SiLU(),
                          nn.Linear(4*d_model, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)                        # [B*S, D]
        logits = self.router(x_flat)                   # [B*S, N]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)           # 归一化

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)                       # 哪些 token 选了 expert i
            token_idx, topk_pos = torch.where(mask)
            if token_idx.numel() > 0:
                out = expert(x_flat[token_idx])
                output.index_add_(0, token_idx, out * weights[token_idx, topk_pos].unsqueeze(-1))
        return output.view(B, S, D)

# 测试
if __name__ == "__main__":
    B, S, D = 2, 16, 512
    x = torch.randn(B, S, D)
    moe = MoE(D, n_experts=8, top_k=2)
    out = moe(x)
    print(f"Input shape: {x.shape}")   # [2, 16, 512]
    print(f"Output shape: {out.shape}") # [2, 16, 512]
```

## 🎯 面试要点

- **Router 决定每个 token 由哪些专家处理**
- **Top-K 路由**: 每个 token 只激活 K 个专家
- **需要 load balancing loss** 防止专家负载不均
- **Mixtral 8x7B**: 8 专家选 2，计算量约 12.9B
- **DeepSeek-V2/V3**: MoE + MLA 实现极致效率
- **面试常问**: MoE 如何减少计算量？如何保证负载均衡？


