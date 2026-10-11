/**
 * 张量流程图 —— `Problem.flowDiagram` 的格式判定。
 *
 * 这个字段原本是自研的行式 DSL（`标签 :: 形状 :: 说明`），由 flowDsl.ts 解析、
 * FlowDiagram.tsx 排版成 SVG。现在改成 mermaid 源码：文档里同一份字符串放进
 * ```mermaid 围栏由 GitHub 原生渲染，站点把同一份字符串交给 mermaid 渲染 ——
 * doc 与 TS 之间仍然只有一份内容，不存在漂移。
 *
 * 迁移期两种格式并存：这里只判断「是不是 mermaid」，不是就退回旧的 DSL 渲染
 * （从冲突上是与 parseFlow 同样的 null 兜底思路），所以 36 题可以分批搬，
 * 没搬完的题目外观一个字都不变。
 */

/** mermaid 流程图的首行：`flowchart TD` / `graph LR`，允许前面有 %%{init}%% 指令行 */
const MERMAID_HEAD = /^(flowchart|graph)\s+(TD|TB|BT|LR|RL)(\s|$)/m;

export function isMermaidFlow(source: string): boolean {
  if (typeof source !== 'string') return false;
  const first = source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '' && !line.startsWith('%%'));
  return first.length > 0 && MERMAID_HEAD.test(first[0]);
}