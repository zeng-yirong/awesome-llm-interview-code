# Hand-torn Code for LLM Interviews

大模型面试手撕代码总结 - 整合两大优质开源仓库，去重统一风格

## 📖 内容

整合自以下两个仓库：
- [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code) ⭐ 879
- [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100) ⭐ 48

涵盖 25 道核心题目，覆盖 13 个类别：
- 👁️ Attention（注意力机制）
- 📏 Normalization（归一化层）
- 📍 Position（位置编码）
- 🔗 FFN（前馈网络）
- 📉 Loss（损失函数）
- ⚙️ Optimizer（优化器）
- 🎮 RL（强化学习）
- 🎯 PEFT（参数高效微调）
- 💾 Efficient（高效训练）
- ⚡ Inference（推理优化）
- 🎲 Sampling（采样策略）
- 🏗️ Architecture（模型架构）
- 📖 Basics（LLM 基础）

## 🚀 本地运行

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 访问 http://localhost:3000
```

## 🌐 GitHub Pages 部署

项目已配置自动化部署，推送到 main 分支后会自动部署到 GitHub Pages。

### 部署步骤

1. **启用 GitHub Pages**
   - 进入仓库 Settings → Pages
   - Source 选择 **GitHub Actions**
   - 保存

2. **推送代码**
   ```bash
   git add .
   git commit -m "deploy to github pages"
   git push origin main
   ```

3. **等待部署完成**
   - GitHub Actions 会自动构建并部署
   - 在 Actions 页面查看部署状态
   - 部署完成后访问：`https://你的用户名.github.io/你的仓库名/`

### 手动触发部署

也可以在 Actions 页面手动触发：
- 进入 Actions → Deploy to GitHub Pages
- 点击 Run workflow

## 📝 特性

- 🔥 **热度标记**：🔥×1~3 表示面试频率（必考/高频/了解）
- ⭐ **难度标记**：⭐×1~5 表示实现难度
- 📐 **原理说明**：每题含思想、公式、流程图
- 💻 **代码实现**：完整可运行的 Python 代码
- 🎯 **面试要点**：关键知识点总结
- 🔍 **搜索筛选**：按分类、热度、难度筛选
- 📊 **排序功能**：按热度、难度、分类排序

## 🛠️ 技术栈

- React 18 + TypeScript
- Vite
- Tailwind CSS
- Framer Motion

## 📄 License

MIT
