import { useEffect, useMemo, useRef, useState } from 'react';
import FlowDiagram from './FlowDiagram';
import { isMermaidFlow } from '../lib/mermaidFlow';
import { useTheme } from '../contexts/ThemeContext';

interface TensorFlowProps {
  source: string;
}

/**
 * 张量流程图。
 *
 * flowDiagram 现在是 mermaid 源码，交给 mermaid 画；同一份字符串也放在文档的
 * ```mermaid 围栏里，GitHub 那边原生渲染成同样一张图。
 *
 * 还没搬完的题目仍是旧的 DSL，交给 FlowDiagram 渲染 —— 迁移期两种格式并存，
 * 未迁移的题目外观不变。
 */
export default function TensorFlow({ source }: TensorFlowProps) {
  if (!isMermaidFlow(source)) return <FlowDiagram source={source} />;
  return <MermaidDiagram source={source} />;
}

/** mermaid 有 1MB 上下，动态 import 让 Vite 把它切成单独的 chunk，首屏不背这个包 */
let mermaidPromise: Promise<typeof import('mermaid').default> | null = null;
function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then((mod) => mod.default);
  }
  return mermaidPromise;
}

function MermaidDiagram({ source }: TensorFlowProps) {
  const { theme } = useTheme();
  const [svg, setSvg] = useState('');
  const [failed, setFailed] = useState(false);
  // mermaid.render 要求全页唯一 id，同一页多张图不能撞
  const idRef = useRef(`tensor-flow-${Math.random().toString(36).slice(2)}`);

  const config = useMemo(
    () => ({
      startOnLoad: false,
      // strict 与 GitHub 一致：标签里的 HTML 只保留 <br/> 这类安全标签
      securityLevel: 'strict' as const,
      theme: theme === 'dark' ? ('dark' as const) : ('default' as const),
      fontFamily: 'inherit',
      flowchart: { curve: 'basis' as const, htmlLabels: true, useMaxWidth: true },
    }),
    [theme],
  );

  useEffect(() => {
    let alive = true;
    setFailed(false);
    loadMermaid()
      .then((mermaid) => {
        mermaid.initialize(config);
        return mermaid.render(idRef.current, source);
      })
      .then(({ svg: out }) => {
        if (alive) setSvg(out);
      })
      .catch(() => {
        // 语法错误 / 渲染失败都退回源码：宁可露出源码，也不要一张空白卡片
        if (alive) setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, [source, config]);

  if (failed) {
    return (
      <pre
        className="text-xs font-mono whitespace-pre p-4 rounded-lg overflow-x-auto"
        style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
      >
        {source}
      </pre>
    );
  }

  return (
    <div
      role="img"
      aria-label="张量流程图"
      className="rounded-lg p-4 overflow-x-auto"
      style={{ backgroundColor: 'var(--bg-tertiary)' }}
      // mermaid 的 SVG 里带 <style> 与 <foreignObject>，只能整段注入（与 CodeBlock 同一手法）
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}