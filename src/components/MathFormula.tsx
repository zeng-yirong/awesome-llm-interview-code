import { useMemo } from 'react';
import katex from 'katex';
import { parseMath } from '../lib/mathBlock';

interface MathFormulaProps {
  source: string;
}

/**
 * 核心公式。
 *
 * formula 是 `$$` 包裹的 LaTeX 时，逐个块交给 KaTeX 排版成 display math。
 * KaTeX 的产物是 HTML + MathML（后者给读屏软件），且颜色取 currentColor，
 * 所以深色主题不需要任何额外覆盖。
 *
 * 解析失败或排版抛错时原样回退到 <pre> —— 36 题可以逐题迁移，未迁移的题目
 * 外观与改动前完全一致。
 *
 * 这里必须是 throwOnError: true：KaTeX 在 throwOnError: false 下「从不抛异常」，
 * 坏公式会渲染成一个红色错误节点，下面的 try/catch 永远进不去，兜底就形同虚设。
 * 宁可整块退回朴素 <pre>，也不要在页面上留一个红框。
 *
 * 这里用 dangerouslySetInnerHTML 是必要且安全的：内容完全由 KaTeX 从 LaTeX 源码
 * 生成，没有任何用户输入进入（与 CodeBlock.tsx 的处理方式一致）。
 */
export default function MathFormula({ source }: MathFormulaProps) {
  const html = useMemo(() => {
    const blocks = parseMath(source);
    if (!blocks) return null;

    try {
      return blocks.map((block) =>
        katex.renderToString(block.latex, {
          displayMode: true,
          throwOnError: true,
          strict: false,
        }),
      );
    } catch {
      // 有任何一个块排不出来，整体退回 <pre>
      return null;
    }
  }, [source]);

  if (!html) {
    return (
      <pre
        className="text-sm font-mono whitespace-pre-wrap p-4 rounded-lg overflow-x-auto"
        style={{
          backgroundColor: 'var(--bg-tertiary)',
          color: 'var(--text-primary)',
        }}
      >
        {source}
      </pre>
    );
  }

  return (
    <div
      className="p-4 rounded-lg space-y-3 overflow-x-auto"
      style={{
        backgroundColor: 'var(--bg-tertiary)',
        color: 'var(--text-primary)',
      }}
    >
      {html.map((blockHtml, index) => (
        <div key={index} dangerouslySetInnerHTML={{ __html: blockHtml }} />
      ))}
    </div>
  );
}