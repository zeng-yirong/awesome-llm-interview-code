# 🧠 Hand-torn Code for LLM Interviews

> **大模型面试手撕代码大全** — 28 道高频题目 · 13 个核心分类 · 双主题支持

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Stars](https://img.shields.io/github/stars/zeng-yirong/awesome-llm-interview-code?style=social)](https://github.com/zeng-yirong/awesome-llm-interview-code)

## ✨ 特性亮点

- 📚 **28 道高频题** — 覆盖 Attention、Normalization、FFN、Loss、PEFT 等核心主题
- 🎨 **双主题** — 白天/黑夜模式一键切换，自动跟随系统偏好
- 🔍 **智能搜索** — 支持搜索题目、原理、公式、要点
- 🏷️ **多维筛选** — 按分类、热度、难度筛选，快速定位
- 📊 **可视化标记** — 🔥 面试热度 + ⭐ 实现难度，优先级一目了然
- 📖 **完整文档** — 每题含原理、公式、流程图、代码、面试要点

---

## 📋 题目总览

> 🔥 = 面试热度（🔥🔥🔥 必考 / 🔥🔥 高频 / 🔥 了解）  
> ⭐ = 实现难度（⭐ 简单 → ⭐⭐⭐⭐⭐ 困难）

### 👁️ Attention（注意力机制）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 1 | [缩放点积注意力](./docs/01-attention/scaled-dot-product-attention.md) | 🔥🔥🔥 | ⭐⭐⭐ |
| 2 | [多头注意力](./docs/01-attention/multi-head-attention.md) | 🔥🔥🔥 | ⭐⭐⭐⭐ |
| 3 | [因果掩码](./docs/01-attention/causal-mask.md) | 🔥🔥🔥 | ⭐⭐ |
| 4 | [分组查询注意力 GQA](./docs/01-attention/gqa.md) | 🔥🔥🔥 | ⭐⭐⭐⭐ |
| 5 | [Flash Attention](./docs/01-attention/flash-attention.md) | 🔥🔥 | ⭐⭐⭐⭐⭐ |
| 6 | [多头潜在注意力 MLA](./docs/01-attention/mla.md) | 🔥🔥 | ⭐⭐⭐⭐⭐ |

### 📏 Normalization（归一化层）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 7 | [层归一化 LayerNorm](./docs/02-normalization/layer-norm.md) | 🔥🔥🔥 | ⭐⭐ |
| 8 | [RMS 归一化](./docs/02-normalization/rms-norm.md) | 🔥🔥🔥 | ⭐⭐ |

### 📍 Position（位置编码）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 9 | [旋转位置编码 RoPE](./docs/03-position/rope.md) | 🔥🔥🔥 | ⭐⭐⭐⭐ |

### 🔗 FFN（前馈网络）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 10 | [标准前馈网络 FFN](./docs/04-ffn/ffn.md) | 🔥🔥 | ⭐⭐ |
| 11 | [SwiGLU](./docs/04-ffn/swiglu.md) | 🔥🔥 | ⭐⭐⭐ |
| 12 | [混合专家模型 MoE](./docs/04-ffn/moe.md) | 🔥🔥 | ⭐⭐⭐⭐ |

### 📉 Loss（损失函数）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 13 | [交叉熵 / LM Loss](./docs/05-loss/cross-entropy.md) | 🔥🔥🔥 | ⭐⭐ |
| 14 | [DPO 损失](./docs/05-loss/dpo.md) | 🔥🔥🔥 | ⭐⭐⭐⭐ |
| 15 | [PPO 损失](./docs/05-loss/ppo.md) | 🔥🔥🔥 | ⭐⭐⭐⭐⭐ |
| 16 | [GRPO 损失](./docs/05-loss/grpo.md) | 🔥🔥🔥 | ⭐⭐⭐⭐ |

### ⚙️ Optimizer（优化器）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 17 | [AdamW 优化器](./docs/06-optimizer/adamw.md) | 🔥🔥🔥 | ⭐⭐⭐ |

### 🎮 RL（强化学习）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 18 | [广义优势估计 GAE](./docs/07-rl/gae.md) | 🔥🔥🔥 | ⭐⭐⭐⭐ |

### 🎯 PEFT（参数高效微调）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 19 | [LoRA](./docs/08-peft/lora.md) | 🔥🔥🔥 | ⭐⭐⭐ |

### 💾 Efficient（高效训练）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 20 | [混合精度训练](./docs/09-efficient/mixed-precision.md) | 🔥🔥 | ⭐⭐⭐ |
| 21 | [梯度检查点](./docs/09-efficient/gradient-checkpointing.md) | 🔥🔥 | ⭐⭐⭐ |

### ⚡ Inference（推理优化）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 22 | [KV Cache](./docs/01-attention/kv-cache.md) | 🔥🔥🔥 | ⭐⭐⭐ |
| 23 | [分页注意力 PagedAttention](./docs/10-inference/paged-attention.md) | 🔥🔥 | ⭐⭐⭐⭐ |
| 24 | [投机解码](./docs/10-inference/speculative-decoding.md) | 🔥🔥 | ⭐⭐⭐⭐ |

### 🎲 Sampling（采样策略）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 25 | [采样策略](./docs/11-sampling/sampling.md) | 🔥🔥🔥 | ⭐⭐ |

### 🏗️ Architecture（模型架构）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 26 | [Decoder-Only 架构](./docs/12-architecture/decoder-only.md) | 🔥🔥🔥 | ⭐⭐⭐ |

### 📖 Basics（LLM 基础）

| # | 题目 | 🔥 | ⭐ |
|:-:|:-----|:-:|:-:|
| 27 | [梯度与反向传播](./docs/13-basics/backprop.md) | 🔥🔥 | ⭐⭐ |
| 28 | [激活函数](./docs/13-basics/activation.md) | 🔥🔥 | ⭐ |

---

## 📂 项目结构

```
awesome-llm-interview-code/
├── docs/                  # 📚 题目文档（28 题）
│   ├── 01-attention/      # 👁️ 注意力机制（6 题）
│   ├── 02-normalization/  # 📏 归一化层（2 题）
│   ├── 03-position/       # 📍 位置编码（1 题）
│   ├── 04-ffn/            # 🔗 前馈网络（3 题）
│   ├── 05-loss/           # 📉 损失函数（4 题）
│   ├── 06-optimizer/      # ⚙️ 优化器（1 题）
│   ├── 07-rl/             # 🎮 强化学习（1 题）
│   ├── 08-peft/           # 🎯 参数高效微调（1 题）
│   ├── 09-efficient/      # 💾 高效训练（2 题）
│   ├── 10-inference/      # ⚡ 推理优化（2 题）
│   ├── 11-sampling/       # 🎲 采样策略（1 题）
│   ├── 12-architecture/   # 🏗️ 模型架构（1 题）
│   └── 13-basics/         # 📖 LLM 基础（2 题）
├── src/                   # 💻 前端源码
│   ├── components/        # React 组件
│   ├── contexts/          # 主题上下文
│   ├── data/              # 题目数据
│   └── App.tsx            # 主应用
├── README.md              # 📖 项目说明
└── DEPLOY.md              # 🚀 部署指南
```

---

## 🏷️ 标记说明

### 🔥 面试热度

| 标记 | 含义 | 复习建议 |
|:----:|:-----|:---------|
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

## 📖 文档结构

每篇题目文档包含以下统一结构：

| 章节 | 内容 |
|:-----|:-----|
| 📌 **原理与思想** | 核心原理的文字描述 |
| 📐 **核心公式** | 关键数学公式 |
| 📊 **张量流程图** | ASCII 流程图展示数据变换过程 |
| 💻 **代码实现** | 完整可运行的 Python 代码 |
| 🎯 **面试要点** | 面试中需要掌握的关键知识点 |

---

## 🚀 快速开始

### 🌐 在线浏览

访问 GitHub Pages 网站：[https://zeng-yirong.github.io/awesome-llm-interview-code/](https://zeng-yirong.github.io/awesome-llm-interview-code/)

### 💻 本地运行

```bash
# 克隆仓库
git clone https://github.com/zeng-yirong/awesome-llm-interview-code.git
cd awesome-llm-interview-code

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 访问 http://localhost:3000
```

### 📦 构建部署

```bash
# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

---

## 🤝 致谢

本项目整合自以下两个优质开源仓库：

- [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) — 大模型面试手撕代码大全
- [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100) — LLM 时代的 Hot 100

感谢原作者的辛勤付出！

---

## 📄 License

MIT
