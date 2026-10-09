# 混合专家模型 (Mixture of Experts, MoE)

> 稀疏激活，大参数量小计算量

## 📌 原理与思想

将 FFN 替换为多个"专家"网络，Router 为每个 token 选择 Top-K 个专家处理。总参数量大但每次只激活部分，计算量可控。

**它解决什么问题**
- 稠密模型里参数和计算是绑死的：想把 7B 变成 70B，FLOPs 也跟着涨约 10 倍。
- 但推理时并非每个 token 都需要全部容量 —— 代码、中文、数学符号该由不同的参数来处理。
- 条件计算把「参数量」和「计算量」解耦：参数量决定容量上限，每个 token 实际激活多少专家决定开销。

**核心思想**
- 把一个大 FFN 切成 `N` 个专家，每个 token 只走 Top-K 个（通常 `K=2`）。
- Router 只是一个 `[D, N]` 的极小的线性层，为每个 token 打分，softmax 后取 Top-K 的权重。
- 总参数是 `N` 份专家，但每个 token 只做 `K` 份专家的矩阵乘法，所以 FLOPs 只有 `K/N`。

**算法步骤与推导**
- 先把 `[B, S, D]` 摊平成 `[B·S, D]` —— 路由是逐 token 的，序列维在这里没有意义。
- Router 给出 `logits: [B·S, N]`，softmax 后取 Top-K，得到专家下标与路由权重。
- 每个专家只处理分到自己的那些 token（gather 成连续块再分组做矩阵乘），算完按权重加权求和。
- 最后 reshape 回 `[B, S, D]` 与残差流对齐；没被选中的专家这一步完全不参与计算。

**对比与代价**
- Mixtral 8x7B：8 个专家选 2 个，46.7B 总参数，但每个 token 实际只算约 12.9B。
- 代价是显存：所有专家都得加载，总参数量一份都不能少，小 batch 推理时性价比尤其差。
- 训练要额外加 load balance loss，否则所有 token 会挤到同一两个专家上，其余专家永远得不到训练。
- 分布式训练的主要瓶颈是通信：专家分散在不同卡上，token 要 all-to-all 发过去再发回来。

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


