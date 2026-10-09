import { useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { parseFlow } from '../lib/flowDsl';
import { layoutFlow, COLOR } from '../lib/flowLayout';

interface FlowDiagramProps {
  source: string;
}

/** 量到宽度之前的首帧用它顶着 —— useLayoutEffect 会在绘制前把真实宽度算出来，看不到跳动 */
const DEFAULT_WIDTH = 640;

/**
 * 张量流程图。
 *
 * flowDiagram 是 DSL 时，交给 flowLayout 算出几何，再渲染成一张真正的 SVG 有向图：
 * 圆角节点 + 带箭头的连线 + 并行分支的扇出/扇入总线。
 * 文字宽度用 canvas 字体度量实测（见 flowLayout.measure），所以折行与盒子高度都是准的，
 * 不依赖字符数估算，也不受系统字体差异影响。
 *
 * 解析失败时原样回退到 <pre> —— 36 题可以分批迁移，未迁移的题目外观与改动前完全一致。
 */
export default function FlowDiagram({ source }: FlowDiagramProps) {
  const doc = useMemo(() => parseFlow(source), [source]);
  const hostRef = useRef<HTMLDivElement>(null);
  const [hostWidth, setHostWidth] = useState(0);

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host || !doc) return;
    const sync = () => setHostWidth(host.clientWidth);
    sync();
    const observer = new ResizeObserver(sync);
    observer.observe(host);
    return () => observer.disconnect();
  }, [doc]);

  const layout = useMemo(
    () => (doc ? layoutFlow(doc, hostWidth || DEFAULT_WIDTH) : null),
    [doc, hostWidth],
  );

  // useId 会产出 ":r1:" 这类带冒号的字符串，放进 url(#…) 里有风险，先洗掉
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '');
  const arrowId = `flow-arrow-${uid}`;

  if (!doc) {
    return (
      <pre
        className="text-xs font-mono whitespace-pre p-4 rounded-lg overflow-x-auto"
        style={{
          backgroundColor: 'var(--bg-tertiary)',
          color: 'var(--text-secondary)',
        }}
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
    >
      {/* 宽度在内层量：外层带 p-4，clientWidth 会把内边距算进来，直接用会永远多出一条横向滚动条 */}
      <div ref={hostRef} style={{ width: '100%' }}>
        {layout && (
          <svg width={layout.width} height={layout.height} style={{ display: 'block' }}>
            <defs>
              <marker
                id={arrowId}
                viewBox="0 0 8 8"
                refX="7"
                refY="4"
                markerWidth="7"
                markerHeight="7"
                markerUnits="userSpaceOnUse"
                orient="auto"
              >
                <path d="M 0 0 L 8 4 L 0 8 Z" fill={COLOR.line} />
              </marker>
            </defs>

            {layout.paths.map((path, index) => (
              <path
                key={`p${index}`}
                d={path.d}
                fill="none"
                stroke={path.stroke}
                strokeWidth={1.5}
                strokeLinecap="round"
                markerEnd={path.arrow ? `url(#${arrowId})` : undefined}
              />
            ))}

            {layout.rects.map((rect, index) => (
              <rect
                key={`r${index}`}
                x={rect.x}
                y={rect.y}
                width={rect.w}
                height={rect.h}
                rx={rect.rx}
                fill={rect.fill}
                stroke={rect.stroke}
              />
            ))}

            {layout.accents.map((accent, index) => (
              <rect
                key={`a${index}`}
                x={accent.x}
                y={accent.y}
                width={accent.w}
                height={accent.h}
                rx={accent.w / 2}
                fill={accent.fill}
              />
            ))}

            {layout.texts.map((text, index) => (
              <text
                key={`t${index}`}
                x={text.x}
                y={text.y}
                fill={text.fill}
                fontStyle={text.italic ? 'italic' : undefined}
                textAnchor={text.anchor ?? 'start'}
                style={{ font: text.font, whiteSpace: 'pre' }}
              >
                {text.text}
              </text>
            ))}
          </svg>
        )}
      </div>
    </div>
  );
}