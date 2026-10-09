# 混合专家模型 (Mixture of Experts, MoE)

> 稀疏激活，大参数量小计算量

## 📌 原理与思想

### 核心概念
将 FFN 替换为多个"专家"网络，Router 为每个 token 选择 Top-K 个专家处理。总参数量大但每次只激活部分，计算量可控。相比 Dense 模型，MoE 可以增加模型容量但不增加计算量，如 Mixtral 8x7B 有 46.7B 参数但实际计算量仅约 12.9B。

### 核心思想
通过 Router 网络为每个 token 动态选择最相关的 K 个专家处理，实现稀疏激活。每个专家是独立的 FFN，只处理分配到的 token，最后加权融合。

### 算法步骤
1. Router 计算：logits = Router(x)，得到每个专家的得分
2. Top-K 选择：选出得分最高的 K 个专家
3. 权重归一化：weights = softmax(top_k_logits)
4. 专家处理：每个 expert 处理分配到的 token
5. 加权融合：output = Σ(weights[i] × expert[i](x))

## 📐 核心公式

$$
\operatorname{MoE}(x)=\sum_{i\in\operatorname{TopK}}\operatorname{softmax}\!\left(\operatorname{Router}(x)\right)_i E_i(x)
$$

$$
\operatorname{Router}(x)=W_rx,\qquad W_r\in\mathbb{R}^{D\times N},\qquad K=2
$$

$$
\text{Mixtral 8x7B:}\quad N=8,\ K=2,\qquad \text{FLOPs}\approx 12.9\text{B}\ \text{vs}\ 46.7\text{B total}
$$

## 📊 张量流程图

```
# 按 token 路由到 Top-K 个专家
x = reshape(x, [B*S, D]) :: [B*S, D] :: 摊平序列维，路由是逐 token 的
logits = Router(x) :: [B*S, N] :: Router 只是一个 [D, N] 的线性层
Top-K :: indices, weights :: 通常 K=2，取出下标与权重
w = softmax(选中的 logits) :: [B*S, K] :: 路由权重，用来加权求和
E_k(x) :: 每个专家都是一个完整的 FFN
y = Σ_k w_k · E_k(x) :: [B*S, D] :: 只算 K 个专家，其余不参与
$ 总参数是 N 份专家，每个 token 只算 K 份 → FLOPs 只占 K/N
> Mixtral 8x7B：8 选 2，46.7B 总参数，单 token 实际计算约 12.9B
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


