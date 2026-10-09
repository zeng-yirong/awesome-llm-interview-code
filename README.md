# 🧠 Hand-torn Code for LLM Interviews

> **大模型面试手撕代码大全** — 36 道高频题目 · 13 个核心分类 · 双主题支持

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Stars](https://img.shields.io/github/stars/zeng-yirong/awesome-llm-interview-code?style=social)](https://github.com/zeng-yirong/awesome-llm-interview-code)

## ✨ 特性亮点

- 📚 **36 道高频题** — 覆盖 Attention、Normalization、FFN、Loss、PEFT 等核心主题
- 🎨 **双主题** — 白天/黑夜模式一键切换，自动跟随系统偏好
- 🔍 **智能搜索** — 支持搜索题目、原理、公式、要点
- 🏷️ **多维筛选** — 按分类、热度、难度筛选，快速定位
- 📊 **可视化标记** — 🔥 面试热度 + ⭐ 实现难度，优先级一目了然
- 📖 **完整文档** — 每题含原理、公式、流程图、代码、面试要点

---

## 📋 题目总览

> 🔥 = 面试热度（🔥🔥🔥 必考 / 🔥🔥 高频 / 🔥 了解）  
> ⭐ = 实现难度（⭐ 简单 → ⭐⭐⭐⭐⭐ 困难）

<table>
  <thead>
    <tr>
      <th align="left">分类</th>
      <th align="center">#</th>
      <th align="left">题目</th>
      <th align="center">🔥热度</th>
      <th align="center">⭐ 难度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7" align="left">👁️ Attention</td>
      <td align="center">1</td>
      <td align="left"><a href="./docs/01-attention/scaled-dot-product-attention.md">缩放点积注意力</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td align="left"><a href="./docs/01-attention/multi-head-attention.md">多头注意力</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td align="left"><a href="./docs/01-attention/causal-mask.md">因果掩码</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td align="center">4</td>
      <td align="left"><a href="./docs/01-attention/gqa.md">分组查询注意力 GQA</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">5</td>
      <td align="left"><a href="./docs/01-attention/flash-attention.md">Flash Attention</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">6</td>
      <td align="left"><a href="./docs/01-attention/mla.md">多头潜在注意力 MLA</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">7</td>
      <td align="left"><a href="./docs/01-attention/sparse-attention.md">稀疏注意力 NSA / DSA</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="2" align="left">📏 Normalization</td>
      <td align="center">8</td>
      <td align="left"><a href="./docs/02-normalization/layer-norm.md">层归一化 LayerNorm</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td align="center">9</td>
      <td align="left"><a href="./docs/02-normalization/rms-norm.md">RMS 归一化</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="1" align="left">📍 Position</td>
      <td align="center">10</td>
      <td align="left"><a href="./docs/03-position/rope.md">旋转位置编码 RoPE</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="3" align="left">🔗 FFN</td>
      <td align="center">11</td>
      <td align="left"><a href="./docs/04-ffn/ffn.md">标准前馈网络 FFN</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td align="center">12</td>
      <td align="left"><a href="./docs/04-ffn/swiglu.md">SwiGLU</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">13</td>
      <td align="left"><a href="./docs/04-ffn/moe.md">混合专家模型 MoE</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="8" align="left">📉 Loss</td>
      <td align="center">14</td>
      <td align="left"><a href="./docs/05-loss/cross-entropy.md">交叉熵 / LM Loss</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td align="center">15</td>
      <td align="left"><a href="./docs/05-loss/dpo.md">DPO 损失</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">16</td>
      <td align="left"><a href="./docs/05-loss/ppo.md">PPO 损失</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">17</td>
      <td align="left"><a href="./docs/05-loss/grpo.md">GRPO 损失</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">18</td>
      <td align="left"><a href="./docs/05-loss/gspo.md">GSPO 组序列策略优化</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">19</td>
      <td align="left"><a href="./docs/05-loss/dapo.md">DAPO 解耦裁剪与动态采样</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">20</td>
      <td align="left"><a href="./docs/05-loss/opd.md">在线策略蒸馏 OPD</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">21</td>
      <td align="left"><a href="./docs/05-loss/self-opd.md">在线自蒸馏 Self-OPD</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="2" align="left">⚙️ Optimizer</td>
      <td align="center">22</td>
      <td align="left"><a href="./docs/06-optimizer/adamw.md">AdamW 优化器</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">23</td>
      <td align="left"><a href="./docs/06-optimizer/muon.md">Muon 优化器</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="1" align="left">🎮 RL</td>
      <td align="center">24</td>
      <td align="left"><a href="./docs/07-rl/gae.md">广义优势估计 GAE</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="1" align="left">🎯 PEFT</td>
      <td align="center">25</td>
      <td align="left"><a href="./docs/08-peft/lora.md">LoRA</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="3" align="left">💾 Efficient</td>
      <td align="center">26</td>
      <td align="left"><a href="./docs/09-efficient/mixed-precision.md">混合精度训练</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">27</td>
      <td align="left"><a href="./docs/09-efficient/gradient-checkpointing.md">梯度检查点</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">28</td>
      <td align="left"><a href="./docs/09-efficient/fp8-blockwise.md">FP8 分块量化训练</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="3" align="left">⚡ Inference</td>
      <td align="center">29</td>
      <td align="left"><a href="./docs/01-attention/kv-cache.md">KV Cache</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">30</td>
      <td align="left"><a href="./docs/10-inference/paged-attention.md">分页注意力 PagedAttention</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">31</td>
      <td align="left"><a href="./docs/10-inference/speculative-decoding.md">投机解码</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="1" align="left">🎲 Sampling</td>
      <td align="center">32</td>
      <td align="left"><a href="./docs/11-sampling/sampling.md">采样策略</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="2" align="left">🏗️ Architecture</td>
      <td align="center">33</td>
      <td align="left"><a href="./docs/12-architecture/decoder-only.md">Decoder-Only 架构</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐</td>
    </tr>
    <tr>
      <td align="center">34</td>
      <td align="left"><a href="./docs/12-architecture/mtp.md">多 Token 预测 MTP</a></td>
      <td align="center">🔥🔥🔥</td>
      <td align="center">⭐⭐⭐⭐</td>
    </tr>
    <tr>
      <td rowspan="2" align="left">📖 Basics</td>
      <td align="center">35</td>
      <td align="left"><a href="./docs/13-basics/backprop.md">梯度与反向传播</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐⭐</td>
    </tr>
    <tr>
      <td align="center">36</td>
      <td align="left"><a href="./docs/13-basics/activation.md">激活函数</a></td>
      <td align="center">🔥🔥</td>
      <td align="center">⭐</td>
    </tr>
  </tbody>
</table>

---

## 📂 项目结构

```
awesome-llm-interview-code/
├── docs/                  # 📚 题目文档（36 题）
│   ├── 01-attention/      # 👁️ 注意力机制（7 题）
│   ├── 02-normalization/  # 📏 归一化层（2 题）
│   ├── 03-position/       # 📍 位置编码（1 题）
│   ├── 04-ffn/            # 🔗 前馈网络（3 题）
│   ├── 05-loss/           # 📉 损失函数（8 题）
│   ├── 06-optimizer/      # ⚙️ 优化器（2 题）
│   ├── 07-rl/             # 🎮 强化学习（1 题）
│   ├── 08-peft/           # 🎯 参数高效微调（1 题）
│   ├── 09-efficient/      # 💾 高效训练（3 题）
│   ├── 10-inference/      # ⚡ 推理优化（3 题）
│   ├── 11-sampling/       # 🎲 采样策略（1 题）
│   ├── 12-architecture/   # 🏗️ 模型架构（2 题）
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
