# 🧠 Hand-torn Code for LLM Interviews

> **大模型面试手撕代码大全** — 整合两大优质开源仓库，去重统一风格
>
> 每题含 **原理 · 公式 · 流程图 · 代码 · 面试要点**，助你高效备战 LLM 面试

[![Stars](https://img.shields.io/github/stars/ckd0817/LLM-Interview-Code?style=social)](https://github.com/ckd0817/LLM-Interview-Code)
[![Stars](https://img.shields.io/github/stars/cdhx/LLM-Code-Hot-100?style=social)](https://github.com/cdhx/LLM-Code-Hot-100)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

---

## 📋 题目总览

> 🔥 = 面试热度（🔥🔥🔥 必考 / 🔥🔥 高频 / 🔥 了解）
> ⭐ = 实现难度（⭐ 简单 → ⭐⭐⭐⭐⭐ 困难）

| # | 题目 | 分类 | 🔥 | ⭐ | 一句话 |
|:-:|:-----|:-----|:-:|:-:|:-------|
| 1 | [缩放点积注意力](./docs/01-attention/scaled-dot-product-attention.md) | Attention | 🔥🔥🔥 | ⭐⭐⭐ | `softmax(QKᵀ/√d)V` — 所有注意力的基础 |
| 2 | [多头注意力](./docs/01-attention/multi-head-attention.md) | Attention | 🔥🔥🔥 | ⭐⭐⭐⭐ | 并行多头 → 拼接 → 输出投影 |
| 3 | [因果掩码](./docs/01-attention/causal-mask.md) | Attention | 🔥🔥🔥 | ⭐⭐ | 下三角矩阵，防止看到未来信息 |
| 4 | [分组查询注意力 GQA](./docs/01-attention/gqa.md) | Attention | 🔥🔥🔥 | ⭐⭐⭐⭐ | 多 Q 头共享 KV 头，LLaMA 2 标配 |
| 5 | [Flash Attention](./docs/01-attention/flash-attention.md) | Attention | 🔥🔥 | ⭐⭐⭐⭐⭐ | 分块计算 + Online Softmax → O(N) 显存 |
| 6 | [KV Cache](./docs/01-attention/kv-cache.md) | Inference | 🔥🔥🔥 | ⭐⭐⭐ | 缓存历史 KV，避免自回归重复计算 |
| 7 | [多头潜在注意力 MLA](./docs/01-attention/mla.md) | Attention | 🔥🔥 | ⭐⭐⭐⭐⭐ | KV 低秩压缩到潜空间，DeepSeek-V2 核心 |
| 8 | [层归一化 LayerNorm](./docs/02-normalization/layer-norm.md) | Normalization | 🔥🔥🔥 | ⭐⭐ | 沿特征维度归一化，Transformer 标配 |
| 9 | [RMS 归一化](./docs/02-normalization/rms-norm.md) | Normalization | 🔥🔥🔥 | ⭐⭐ | 去掉均值中心化，LLaMA/Mistral 标配 |
| 10 | [旋转位置编码 RoPE](./docs/03-position/rope.md) | Position | 🔥🔥🔥 | ⭐⭐⭐⭐ | 旋转 Q/K 向量注入位置信息 |
| 11 | [标准前馈网络 FFN](./docs/04-ffn/ffn.md) | FFN | 🔥🔥 | ⭐⭐ | 两层 MLP，占 Transformer 2/3 参数量 |
| 12 | [SwiGLU](./docs/04-ffn/swiglu.md) | FFN | 🔥🔥 | ⭐⭐⭐ | 门控 + SiLU 激活，LLaMA/PaLM 标配 |
| 13 | [混合专家模型 MoE](./docs/04-ffn/moe.md) | FFN | 🔥🔥 | ⭐⭐⭐⭐ | 稀疏激活，大参数量小计算量 |
| 14 | [交叉熵 / LM Loss](./docs/05-loss/cross-entropy.md) | Loss | 🔥🔥🔥 | ⭐⭐ | 下一个 token 预测，LLM 训练基础 |
| 15 | [DPO 损失](./docs/05-loss/dpo.md) | Loss | 🔥🔥🔥 | ⭐⭐⭐⭐ | 无需奖励模型，直接优化偏好数据 |
| 16 | [PPO 损失](./docs/05-loss/ppo.md) | Loss | 🔥🔥🔥 | ⭐⭐⭐⭐⭐ | 截断重要性采样比率，RLHF 核心 |
| 17 | [GRPO 损失](./docs/05-loss/grpo.md) | Loss | 🔥🔥🔥 | ⭐⭐⭐⭐ | 去掉 Critic，组内归一化优势 |
| 18 | [AdamW 优化器](./docs/06-optimizer/adamw.md) | Optimizer | 🔥🔥🔥 | ⭐⭐⭐ | 解耦权重衰减，LLM 训练标配 |
| 19 | [广义优势估计 GAE](./docs/07-rl/gae.md) | RL | 🔥🔥🔥 | ⭐⭐⭐⭐ | 偏差-方差折衷的 λ-return |
| 20 | [LoRA](./docs/08-peft/lora.md) | PEFT | 🔥🔥🔥 | ⭐⭐⭐ | ΔW = BA，低秩分解高效微调 |
| 21 | [混合精度训练](./docs/09-efficient/mixed-precision.md) | Efficient | 🔥🔥 | ⭐⭐⭐ | FP16/BF16 计算 + FP32 主权重 |
| 22 | [梯度检查点](./docs/09-efficient/gradient-checkpointing.md) | Efficient | 🔥🔥 | ⭐⭐⭐ | 时间换空间，重新计算代替存储激活 |
| 23 | [分页注意力 PagedAttention](./docs/10-inference/paged-attention.md) | Inference | 🔥🔥 | ⭐⭐⭐⭐ | vLLM 核心，分页管理 KV Cache |
| 24 | [投机解码](./docs/10-inference/speculative-decoding.md) | Inference | 🔥🔥 | ⭐⭐⭐⭐ | 小模型草稿，大模型验证，加速 2-3x |
| 25 | [采样策略](./docs/11-sampling/sampling.md) | Sampling | 🔥🔥🔥 | ⭐⭐ | Temperature / Top-k / Top-p |
| 26 | [Decoder-Only 架构](./docs/12-architecture/decoder-only.md) | Architecture | 🔥🔥🔥 | ⭐⭐⭐ | 因果注意力 + 自回归，现代 LLM 标配 |
| 27 | [梯度与反向传播](./docs/13-basics/backprop.md) | Basics | 🔥🔥 | ⭐⭐ | 链式法则，深度学习的基石 |
| 28 | [激活函数](./docs/13-basics/activation.md) | Basics | 🔥🔥 | ⭐ | ReLU / GELU / SiLU 及其梯度 |

---

## 📂 目录结构

```
docs/
├── 01-attention/          # 👁️ 注意力机制
│   ├── scaled-dot-product-attention.md
│   ├── multi-head-attention.md
│   ├── causal-mask.md
│   ├── gqa.md
│   ├── flash-attention.md
│   ├── kv-cache.md
│   └── mla.md
├── 02-normalization/      # 📏 归一化层
│   ├── layer-norm.md
│   └── rms-norm.md
├── 03-position/           # 📍 位置编码
│   └── rope.md
├── 04-ffn/                # 🔗 前馈网络
│   ├── ffn.md
│   ├── swiglu.md
│   └── moe.md
├── 05-loss/               # 📉 损失函数
│   ├── cross-entropy.md
│   ├── dpo.md
│   ├── ppo.md
│   └── grpo.md
├── 06-optimizer/          # ⚙️ 优化器
│   └── adamw.md
├── 07-rl/                 # 🎮 强化学习
│   └── gae.md
├── 08-peft/               # 🎯 参数高效微调
│   └── lora.md
├── 09-efficient/          # 💾 高效训练
│   ├── mixed-precision.md
│   └── gradient-checkpointing.md
├── 10-inference/          # ⚡ 推理优化
│   ├── paged-attention.md
│   └── speculative-decoding.md
├── 11-sampling/           # 🎲 采样策略
│   └── sampling.md
├── 12-architecture/       # 🏗️ 模型架构
│   └── decoder-only.md
└── 13-basics/             # 📖 LLM 基础
    ├── backprop.md
    └── activation.md
```

---

## 🏷️ 标记说明

### 🔥 面试热度

| 标记 | 含义 | 建议 |
|:----:|:-----|:-----|
| 🔥🔥🔥 | **必考** | 必须熟练手写，理解每个细节 |
| 🔥🔥 | **高频** | 需要掌握核心实现和原理 |
| 🔥 | **了解** | 理解思想，能口述关键步骤 |

### ⭐ 实现难度

| 标记 | 含义 | 预估时间 |
|:----:|:-----|:--------:|
| ⭐ | 简单 | 5 分钟 |
| ⭐⭐ | 较简单 | 10 分钟 |
| ⭐⭐⭐ | 中等 | 15-20 分钟 |
| ⭐⭐⭐⭐ | 较难 | 20-30 分钟 |
| ⭐⭐⭐⭐⭐ | 困难 | 30+ 分钟 |

---

## 📖 每篇文档结构

每个题目的文档都包含以下统一结构：

```markdown
# 题目名称

> 一句话描述

## 📌 原理与思想
核心原理的文字描述

## 📐 核心公式
关键数学公式

## 📊 张量流程图
ASCII 流程图展示数据变换过程

## 💻 代码实现
完整可运行的 Python 代码

## 🎯 面试要点
面试中需要掌握的关键知识点
```

---

## 🚀 快速开始

### 在线浏览

- 🌐 **GitHub Pages**: [在线网站](https://你的用户名.github.io/你的仓库名/)
- 📖 **直接阅读**: 点击上方表格中的题目链接

### 本地运行

```bash
# 克隆仓库
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 安装依赖 & 启动
npm install
npm run dev

# 访问 http://localhost:3000
```

---

## 🤝 致谢

本项目整合自以下两个优质开源仓库：

- [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) — 大模型面试手撕代码大全（⭐879）
- [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100) — LLM 时代的 Hot 100（⭐48）

感谢原作者的辛勤付出！

---

## 📄 License

MIT
