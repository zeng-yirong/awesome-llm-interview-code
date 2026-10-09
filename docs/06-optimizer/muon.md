# Muon 优化器 (Momentum Orthogonalized by Newton-Schulz)

> 动量矩阵先正交化再更新，隐藏层权重的谱范数几何

## 📌 原理与思想

### 核心概念
Muon 只用于二维隐藏层权重：累积动量后，用 Newton-Schulz 迭代把动量矩阵近似成正交矩阵（极分解 `UVᵀ`）作为更新方向，让所有奇异方向等步长；嵌入层、输出头和所有一维参数仍交给 AdamW。规模化时 Muon 会把注意力 logit 推到爆炸，Kimi K2 用 QK-Clip 把 Q/K 权重乘 `√γ`（`γ = τ/S_max`）从源头压住。代价是每步多 5 次矩阵乘法，换来约 2× token 效率，正则与 RMS 对齐后 Adam 的超参可以直接迁移。

### 核心思想
在谱范数几何下做最速下降，最优更新方向是动量矩阵的极分解 `UVᵀ` —— 一个所有奇异值都等于 1 的半正交矩阵。直观说就是只留方向、抹掉幅度：每个奇异方向走同样大的一步，幅度交给学习率统一控制。求极分解不必真做 SVD（慢且数值敏感），Newton-Schulz 迭代只用矩阵乘法就能逼近。

### 算法步骤
1. 参数分组：只有 ≥2 维的隐藏层权重交给 Muon，embedding、lm_head 和所有 1D 参数（bias、norm）仍用 AdamW。
2. 累积动量 → 除以 F 范数归一化（保证谱范数 ≤ 1，否则迭代发散）→ 5 次 NS 迭代 → `θ ← θ - η·O`。
3. 5 步后奇异值落在约 `[0.5, 1.5]` 就够用：系数 `(3.4445, -4.7750, 2.0315)` 正是为「5 步内尽量压平」调出来的。

## 📐 核心公式

```
Muon 更新（只对 ≥ 2 维的隐藏层权重）:
  M_t = μ·M_{t-1} + G_t                  # 累积动量
  O_t = NS5(M_t)                          # 正交化，逼近极分解 UVᵀ
  θ_t = θ_{t-1} - η·O_t

Newton-Schulz 五次迭代（求零次幂）:
  X_0 = G / (‖G‖_F + ε)                   # 先归一化，保证谱范数 ≤ 1
  重复 5 次:
      A = X·Xᵀ
      X ← a·X + (b·A + c·A²)·X
  系数 (a, b, c) = (3.4445, -4.7750, 2.0315)

MuonClip = Muon + QK-Clip（Kimi K2 的做法）:
  S_max = 每个 head 在本 batch 上的最大注意力 logit
  若 S_max > τ:   γ = τ / S_max
                 W_q ← √γ · W_q ,   W_k ← √γ · W_k
  (典型 τ = 100)
```

其中:
- `M_t`: 动量矩阵，形状与权重相同
- `NS5(·)`: 5 次 Newton-Schulz 迭代，逼近 `UVᵀ`。5 次之后奇异值落在约 `[0.5, 1.5]`，不是严格正交，但实践中足够
- `μ`: 动量系数，典型 `0.95`
- `QK-Clip`: 从源头缩放 Q/K 投影权重来压住 logit 增长。用 `√γ` 是因为 logit 正比于 `W_q·W_kᵀ`，放缩一次权重，logit 就被放缩 γ 倍
- 规模化时 Muon 会把注意力 logit 推到爆炸（远超正常量级），QK-Clip 是 Kimi K2 能稳定训到万亿参数的关键

## 📊 张量流程图

```
# 参数分组：Muon 只吃 2D 隐藏层权重
+ emb / lm_head / bias / norm :: 交给 AdamW
+ attn & mlp 的 2D 权重矩阵 :: 交给 Muon

# Muon 单步
G :: [m, n] :: 本步梯度
M = μM + G :: [m, n] :: 累积动量（Nesterov 变体用 G + μ·M）
X = M / (‖M‖_F + ε) :: [m, n] :: 归一化，保证谱范数 ≤ 1，否则迭代发散
NS5 :: 重复 5 次 X ← a·X + (b·A + c·A²)·X，其中 A = X·Xᵀ
O :: [m, n] :: 奇异值 ≈ 1，逼近极分解 UVᵀ
$ θ ← θ - η·O：每个奇异方向走同样大的一步
> 系数 (a, b, c) = (3.4445, -4.7750, 2.0315)，5 步后奇异值落在约 [0.5, 1.5]

# MuonClip 额外一步（周期性执行）
S_max :: 每个 head 在本 batch 上的最大注意力 logit
! S_max > τ → γ = τ / S_max，W_q ← √γ·W_q，W_k ← √γ·W_k
> 典型 τ = 100。logit ∝ W_q·W_kᵀ，缩放一次权重等于缩放 γ 倍 logit，所以取 √γ
```

## 💻 代码实现

```python
import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    用 Newton-Schulz 迭代近似 G 的极分解 UVᵀ（即「零次幂」）
    G: [m, n] 动量矩阵；返回同形状矩阵，奇异值趋近 1
    """
    assert G.ndim == 2, "Muon 只处理二维矩阵参数"
    a, b, c = 3.4445, -4.7750, 2.0315        # 调好的五次多项式系数

    X = G.to(torch.bfloat16)
    # 归一化: 保证谱范数 ≤ 1，否则迭代会发散
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.mT                          # [m, m]
        B = b * A + c * (A @ A)               # 五次多项式
        X = a * X + B @ X                     # [m, n]

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """只用于 2D 隐藏层权重的 Muon 优化器"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                # Nesterov: 用 g + μ·buf 作为动量方向
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                # 先正交化，再更新
                p.add_(zeropower_via_newtonschulz5(update, group["ns_steps"]),
                       alpha=-group["lr"])


@torch.no_grad()
def qk_clip_(model, tau=100.0):
    """
    MuonClip 的 QK-Clip: 从源头压住注意力 logit 的增长
    约定: 模块把本 batch 每 head 的最大 logit 存在 qk_max_logit 属性上
    """
    for module in model.modules():
        s_max = getattr(module, "qk_max_logit", None)
        if s_max is None or s_max <= tau:
            continue
        gamma = tau / s_max                       # 缩放因子
        scale = gamma ** 0.5                      # logit ∝ W_q·W_kᵀ，所以取 √γ
        module.q_proj.weight.mul_(scale)
        module.k_proj.weight.mul_(scale)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. 正交化效果: 奇异值应该都趋近 1
    G = torch.randn(64, 32)
    O = zeropower_via_newtonschulz5(G)
    sv = torch.linalg.svdvals(O.float())
    print("原始奇异值范围:", round(torch.linalg.svdvals(G).min().item(), 3),
          "~", round(torch.linalg.svdvals(G).max().item(), 3))
    print("正交化后范围:", round(sv.min().item(), 3), "~", round(sv.max().item(), 3))

    # 2. Muon 走一步
    w = torch.nn.Parameter(torch.randn(64, 32))
    opt = Muon([w], lr=0.02)
    w.grad = torch.randn(64, 32)
    opt.step()
    print("Muon 一步更新完成，权重形状:", tuple(w.shape))
```

## 🎯 面试要点

- **为什么正交化**: 把动量矩阵的所有奇异值归一化为 1，让更新在每个奇异方向上等步长 —— 对应谱范数下的最速下降
- **为什么 5 步够**: Newton-Schulz 收敛很快，5 步后奇异值已经落在 `[0.5, 1.5]`；继续迭代收益很小而算力翻倍
- **bf16 能跑**: 迭代全是矩阵乘法，没有 SVD 那样的数值敏感操作，bfloat16 下稳定
- **Muon 只管 2D 权重**: 嵌入、输出头、所有 1D 参数（bias、norm）留给 AdamW —— 混合优化器是标准做法
- **Adam 与 Muon 的几何差别**: Adam 是逐坐标自适应步长，完全忽略权重矩阵的行列结构；Muon 在谱范数几何下让每个奇异方向等步长
- **规模化会炸 logit**: Muon 训到万亿参数时注意力 logit 会涨到远超正常量级（上千），QK-Clip 用 `√γ` 缩放 Q/K 权重从源头压住，且它是**无损**的（只维持数值稳定，不改变表达力）
- **为什么是 √γ**: logit 正比于 `W_q·W_kᵀ`，放缩一次权重，logit 就被放缩 γ 倍，所以权重取 √γ
- **QK-Clip 会自己退场**: 训练早期生效，模型稳定后 `S_max` 不再超阈值，机制自动不再触发
- **实测收益**: 相比 AdamW 约有 2× token 效率；正则与 RMS 对齐后 Adam 的超参可以直接迁移
- **面试常问**: Muon 和 Adam 的几何差别是什么？为什么 QK-Clip 用 √γ 而不是 γ？