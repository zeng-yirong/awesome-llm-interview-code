# FP8 分块量化训练 (Blockwise FP8 Training)

> 按 128 一块算缩放因子，FP8 训练的关键全在块级缩放

## 📌 原理与思想

训练用 FP8（前向的激活和权重用 e4m3，反向的梯度用 e5m2）能省显存带宽并提高吞吐，但整张量共用一个缩放因子会被少数离群值拖垮——为了覆盖极值，绝大多数正常值被迫压到很低的精度。解决办法是分块：每 128 个元素（或 128×128 的块）单独算 amax 和缩放因子，块内共享一个 scale。矩阵乘法在 FP8 上做，累加仍在高精度。

**它解决什么问题**
- FP8 只占 1 字节，显存和带宽相对 FP16 再减半，是最直接的吞吐手段。
- 但它的动态范围极窄：e4m3 只有 3 位尾数，可表示的最大值只有 `±448`。
- 而激活值的分布是重尾的，个别通道的幅度能比中位数大几个数量级；整张量共用一个缩放因子时，为了装下离群值，正常值全被压到只剩几档可表示，量化噪声巨大。

**核心思想**
- 缩放因子不该整张量共用一个，而应该按块各算各的：每 128 个元素单独求 `amax` 定 scale。
- 这样离群值的影响被限制在它自己那一块，其余块按自己的分布正常量化。
- 矩阵乘法在 FP8 上做，但累加提到高精度 —— 精度损失只发生在「存」，不发生在「算」。

**算法步骤与推导**
- `scale = amax(block) / 448`，`448` 是 e4m3 的可表示上界，除完正好用满整个范围。
- 量化 `x_q = clamp(x/scale, -448, 448).to(fp8)`，只保留 3 位尾数；反量化 `x̂ = x_q·scale`。
- 缩放因子可以提到块外：`y = Σ_block s_x·s_w·(x_q·w_q)`，FP8 相乘、高精度累加，缩放因子不进乘法开销。
- DeepSeek-V3 的配置：激活用 `1×128` 的块（per-token per-128-channel），权重用 `128×128` 的块。

**对比与代价**
- 相对朴素的整张量缩放：正常值的有效位数不再被离群值挤掉，量化误差大幅下降。
- 相对 FP16：显存与带宽再减半；代价是需要专门支持 FP8 的硬件（Hopper 及以后）。
- 代价是每块都要算一次 `amax`，多了归约与 scale 管理的开销；块越小精度越好，但元数据占比也越高。

## 📐 核心公式

```
缩放因子（块内 amax）:
  scale = amax(block) / 448                # 448 是 e4m3 的可表示上界

量化 / 反量化:
  x_q = clamp(x / scale, -448, 448).to(fp8)      # 只保留 3 位尾数
  x̂  = x_q · scale

块级矩阵乘法（缩放因子提到块外）:
  y = Σ_block (x_q · s_x) · (w_q · s_w)
    = Σ_block s_x · s_w · (x_q · w_q)       # FP8 乘 + 高精度累加

DeepSeek-V3 的配置:
  激活: 1×128 的块（per-token per-128-channel）
  权重: 128×128 的块
  累加: 提升到高精度（FP32 / 张量核内高精度累加）
```

其中:
- `e4m3`: 1 符号 + 4 指数 + 3 尾数，动态范围小但精度高 → 用于前向的激活和权重
- `e5m2`: 1 符号 + 5 指数 + 2 尾数，动态范围大但精度低 → 用于反向的梯度
- `448`: e4m3 的最大可表示值（e5m2 是 57344）
- 累加必须高精度：FP8 乘积本身的精度就低，累加若也用 FP8 会迅速累积误差

## 📊 张量流程图

```
# 整张量共用一个 scale：离群值把所有正常值拖下水
x :: [4096] :: 含离群值 1200
scale = amax / 448 :: 2.68 :: 为了装下 1200，只能把范围拉到最大
! ✗ 正常值 ~1.0 除以 2.68 只剩 0.37，3 位尾数不够用，精度崩了
# 块级缩放：每块自己算 amax
+ block_0 :: amax = 1.2 → s = 0.0027 :: 块内共享一个 scale
+ block_1 :: amax = 0.9 → s = 0.0020
+ block_7 :: amax = 1200 → s = 2.68 :: 离群值只影响自己这一块
x_q = (x / s).to(fp8_e4m3) :: 1 字节/元素 :: 只保留 3 位尾数
GEMM :: Σ s_x·s_w·(x_q @ w_q) :: FP8 相乘、高精度累加
$ 显存与带宽相对 FP16 再减半，累加精度不变
> DeepSeek-V3：激活 1×128 分块，权重 128×128 分块
```

## 💻 代码实现

```python
import torch

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def per_block_quant(x, block_size=128, fmt="e4m3"):
    """
    按最后一维分块做 FP8 量化
    x: [..., N]；block_size: 每块元素数
    返回: (量化再反量化后的值, 每块的缩放因子)
    注意: 需要 torch >= 2.1 才有 float8_e4m3fn
    """
    assert x.shape[-1] % block_size == 0, "最后一维要能被 block_size 整除"
    orig_shape = x.shape
    n_blocks = orig_shape[-1] // block_size
    xb = x.reshape(-1, n_blocks, block_size)                     # [M, n_blocks, B]

    # 1. 每块一个 amax → 一个缩放因子（clamp_min 防止除零）
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)   # [M, n_blocks, 1]
    scale = amax / FP8_MAX[fmt]

    # 2. 量化到 FP8 再反量化（模拟「FP8 存、高精度用」）
    x_q = (xb / scale).clamp(-FP8_MAX[fmt], FP8_MAX[fmt]).to(FP8_DTYPE[fmt])
    x_hat = x_q.to(x.dtype) * scale

    return x_hat.reshape(orig_shape), scale.reshape(-1, n_blocks)


def fp8_gemm_sim(x, w, block_size=128):
    """
    模拟 FP8 块级矩阵乘法: FP8 乘 + 高精度累加
    x: [M, K]   w: [K, N]
    """
    # 激活按 1×128 的块量化（沿 K 维切块）
    x_hat, _ = per_block_quant(x, block_size, fmt="e4m3")
    # 权重按 128×128 的块量化
    w_hat, _ = per_block_quant(w.t().contiguous(), block_size, fmt="e4m3")

    # 真实实现是 FP8 张量核 + 高精度累加；这里用反量化后的值算等价的数学结果
    return x_hat @ w_hat.t().contiguous()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256, 512)
    x[0, 0] = 1200.0                       # 人为插一个离群值

    # 对比: 整张量一个 scale  vs  分块 scale
    naive_scale = x.abs().amax() / FP8_MAX["e4m3"]
    x_naive = (x / naive_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    x_naive = x_naive.to(x.dtype) * naive_scale
    x_block, _ = per_block_quant(x, 128, "e4m3")

    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    print("朴素量化相对误差:", round(rel(x_naive, x), 5))
    print("分块量化相对误差:", round(rel(x_block, x), 5))

    w = torch.randn(512, 128)
    print("FP8 GEMM 输出形状:", tuple(fp8_gemm_sim(x, w).shape))
```

## 🎯 面试要点

- **核心一句话**: FP8 训练的成败在缩放粒度，块级缩放把离群值的影响关在块内
- **为什么必须分块**: FP8 的动态范围很窄（e4m3 只有 3 位尾数、上界 448），而激活分布是重尾的；整张量共享 scale 时，为了装下离群值，正常值全跌到只剩几档可表示
- **e4m3 与 e5m2 的分工**: 前向用 e4m3（要精度），反向梯度用 e5m2（要动态范围）
- **累加精度不能省**: FP8 只用来做乘，累加必须回到高精度，否则误差快速累积
- **块大小的权衡**: 块越小，离群值影响越小但缩放因子的存储和计算开销越大；`1×128`（激活）和 `128×128`（权重）是常见折中
- **必须让缩放因子以可融合的方式参与**: 真实实现把 scale 融进张量核的 epilogue，避免额外的反量化访存
- **和 BF16 混合精度的区别**: BF16 混合精度靠「主权重 FP32 + 计算 BF16」保精度；FP8 靠「块级缩放 + 高精度累加」保精度，省的是带宽
- **面试常问**: 为什么 FP8 一定要分块？为什么累加不能用 FP8？e4m3 和 e5m2 各自用在哪一侧？