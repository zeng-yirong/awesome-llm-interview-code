/**
 * 核心公式 —— `Problem.formula` 的解析器。
 *
 * 背景：formula 原本是纯文本伪数学（`L = -Σ log P(xₜ | x<t)`），既排不出分式与上下标，
 * 也没法在文档里渲染。改成 LaTeX 后站点用 KaTeX 排版，文档里 GitHub 原生渲染 `$$`，
 * 而两边共用同一个字符串 —— doc 与 TS 之间的内容漂移由构造消失。
 *
 * 格式（`$$` 独占一行，块外不得有文字）：
 *
 *   $$
 *   \text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
 *   $$
 *
 * 一题可以有多个块，块与块之间空一行（如 DAPO 的多项改动、稀疏注意力的两套方案）。
 * 块内不含中文：中文在 KaTeX 里要走字体回退、在 MathJax 里排版也不稳，需要名字时
 * 用英文的 `\text{}`（GitHub 的数学渲染器会拒掉 `\operatorname`，见 scripts/check-flow.mjs）。
 *
 * 只要没有 `$$`、`$$` 未闭合、或块外出现非空行，就返回 null；调用方回退到旧的 <pre>。
 * 因此迁移可以逐题进行，网站不会中途损坏。
 */

/** 一个 display math 块 */
export interface MathBlock {
  latex: string;
}

const DELIM = '$$';

/**
 * 把 formula 解析成若干个 LaTeX 块；不是新格式就返回 null（调用方走 <pre> 兜底）。
 */
export function parseMath(source: string): MathBlock[] | null {
  if (typeof source !== 'string' || source.trim() === '') return null;

  const blocks: MathBlock[] = [];
  // null 表示当前不在 `$$` 块内
  let buffer: string[] | null = null;

  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (line === DELIM) {
      if (buffer === null) {
        buffer = [];
      } else {
        const latex = buffer.join('\n').trim();
        if (latex === '') return null;
        blocks.push({ latex });
        buffer = null;
      }
      continue;
    }

    if (buffer !== null) {
      buffer.push(rawLine.trim());
      continue;
    }

    // 块外只允许空行 —— 「只保留纯数学」
    if (line !== '') return null;
  }

  if (buffer !== null) return null; // `$$` 没闭合
  if (blocks.length === 0) return null;

  return blocks;
}

/**
 * 纯排版 / 结构类命令，本身不是可搜索的词。留着的话搜 "operator"、"left"、"begin"
 * 会命中几乎所有题目，把搜索结果淹掉。其余命令名一律保留 —— 搜 "epsilon"、"softmax"
 * 都该命中，`\text{…}` / `\operatorname{…}` 花括号里的内容也会照常留下。
 */
const STRUCTURAL_MACROS = new Set([
  'operatorname', 'mathrm', 'mathbf', 'mathcal', 'mathbb', 'mathfrak', 'mathsf', 'mathtt', 'boldsymbol',
  'text', 'textrm', 'textbf', 'textit',
  'left', 'right', 'bigl', 'bigr', 'Bigl', 'Bigr', 'biggl', 'biggr', 'Biggl', 'Biggr',
  'frac', 'dfrac', 'tfrac', 'sqrt',
  'begin', 'end', 'aligned', 'array', 'cases', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix', 'split',
  'quad', 'qquad', 'displaystyle', 'limits', 'nolimits', 'substack',
  'underbrace', 'overbrace', 'phantom', 'hspace', 'nonumber', 'notag',
]);

/**
 * 供搜索用的纯文本投影：剥掉 `$$` 与 LaTeX 语法，但保留命令名本身的文字，
 * 这样 `\operatorname{softmax}`、`\frac`、`\epsilon` 仍然能命中 "softmax" / "epsilon"。
 * 不是新格式时原样返回，旧文本的搜索行为不受影响。
 */
export function plainMath(source: string): string {
  const blocks = parseMath(source);
  if (!blocks) return source;

  return blocks
    .map((block) => block.latex)
    .join(' ')
    // `\begin{aligned}` / `\end{cases}` 的容器名不是词：留着会让 "aligned" 命中每一道多行公式
    .replace(/\\(?:begin|end)\{[a-zA-Z*]+\}/g, ' ')
    // `\\[2pt]` 只是行距，同理
    .replace(/\\\\\s*\[[^\]]*\]/g, ' ')
    .replace(/\\([a-zA-Z]+)/g, (_, name: string) => (STRUCTURAL_MACROS.has(name) ? ' ' : ` ${name} `))
    .replace(/[{}^_&$\\]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}