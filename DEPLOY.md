# GitHub Pages 部署指南

## ✅ 已完成的修复

1. **添加 SPA 路由支持** - 创建 `public/404.html` 和修改 `index.html`
2. **优化 vite.config.js** - 使用 `base: './'` 适配 GitHub Pages
3. **GitHub Actions 自动部署** - 推送到 main 分支自动部署

---

## 🚀 部署步骤

### 第一步：推送代码到 GitHub

```bash
# 添加所有修改
git add .

# 提交
git commit -m "fix: 修复 GitHub Pages 404 问题，添加 SPA 路由支持"

# 推送到 main 分支
git push origin main
```

### 第二步：配置 GitHub Pages

1. 打开你的 GitHub 仓库页面
2. 点击 **Settings**（设置）
3. 左侧菜单找到 **Pages**
4. 在 **Source** 部分：
   - 选择 **GitHub Actions**（不是 "Deploy from a branch"）
5. 保存

### 第三步：等待自动部署

1. 推送代码后，GitHub Actions 会自动触发
2. 在仓库的 **Actions** 页面查看部署进度
3. 等待约 1-2 分钟
4. 部署完成后，访问链接会显示在 Actions 页面

---

## 🔍 访问地址

部署成功后，你的网站地址是：

```
https://zeng-yirong.github.io/awesome-llm-interview-code/
```

仓库信息：
- 用户名：`zeng-yirong`
- 仓库名：`awesome-llm-interview-code`
- 访问地址：`https://zeng-yirong.github.io/awesome-llm-interview-code/`

---

## ❗ 常见问题

### 问题 1：访问显示 404

**原因**：GitHub Pages 的 Source 设置不正确

**解决**：
1. 进入 Settings → Pages
2. 确保 Source 选择的是 **GitHub Actions**
3. 不是 "Deploy from a branch"

### 问题 2：页面空白或资源加载失败

**原因**：`base` 配置不正确

**解决**：
1. 检查 `vite.config.js` 中的 `base` 配置
2. 项目站点应该是 `base: './'` 或 `base: '/仓库名/'`
3. 用户站点（username.github.io）应该是 `base: '/'`

### 问题 3：刷新页面 404

**原因**：GitHub Pages 不支持 SPA 路由

**解决**：
- 已添加 `public/404.html` 和 `index.html` 中的路由恢复脚本
- 确保这两个文件都已推送到仓库

### 问题 4：Actions 部署失败

**检查**：
1. 进入 Actions 页面查看错误日志
2. 常见错误：
   - Node 版本问题 → 已配置 Node 20
   - npm install 失败 → 检查 package.json
   - 构建失败 → 本地运行 `npm run build` 测试

---

## 📋 验证清单

部署前请确认：

- [ ] `vite.config.js` 中 `base: './'`
- [ ] `public/404.html` 文件存在
- [ ] `index.html` 中包含 SPA 路由恢复脚本
- [ ] `.github/workflows/deploy.yml` 文件存在
- [ ] 代码已推送到 main 分支
- [ ] GitHub Pages Source 设置为 "GitHub Actions"

---

## 🔄 手动触发部署

如果自动部署没有触发，可以手动触发：

1. 进入仓库的 **Actions** 页面
2. 左侧选择 **Deploy to GitHub Pages**
3. 点击 **Run workflow**
4. 选择分支（main），点击 **Run workflow**

---

## 📞 获取帮助

如果仍然遇到问题：

1. 检查 Actions 页面的部署日志
2. 查看浏览器控制台的错误信息
3. 确认仓库名称和用户名正确
4. 等待 5 分钟后刷新页面（GitHub Pages 有缓存）

---

## 🎯 快速测试

部署成功后，测试以下功能：

1. ✅ 首页正常加载
2. ✅ 搜索功能正常
3. ✅ 题目卡片可以展开/收起
4. ✅ 代码高亮正常显示
5. ✅ 复制代码功能正常
6. ✅ 刷新页面不会 404

---

**祝部署顺利！** 🎉
