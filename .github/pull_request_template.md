<!-- 一道题一个 PR；多个笔误的批量修正可以合并。详见 CONTRIBUTING.md -->

## 这个 PR 做了什么

<!-- 一句话说清：新增题目 / 纠错 / 改进表述 -->

关联 issue：<!-- #123，没有就删掉这行 -->

---

## 如果这是新增题目

- [ ] 文档已写：`docs/<分类>/<slug>.md`，5 个 H2 小节标题与既有文档一字不差
- [ ] `src/data/problems.ts` 已加条目，内容是从文档**复制**过去的（不是各写一遍）
- [ ] `scripts/check-flow.mjs` 的 `DOC_OF` 已注册 `'<id>': 'docs/...md'`
- [ ] `README.md` 总览表已加行，该分类的 `rowspan` 数字已 +1
- [ ] 若是新分类：`Category` union 与 `docs/` 新目录都已加上

## 所有 PR

- [ ] 本地 `npm run check` 通过
- [ ] 本地 `npm run typecheck` 通过
- [ ] 若用了新的 LaTeX 宏：已在 GitHub 上目视确认公式渲染正常
- [ ] 没有改动与本次贡献无关的文件（尤其是 `main` 上已有的题目内容）