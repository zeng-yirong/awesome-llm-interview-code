import type { FlowDoc, FlowSegment, Tone } from './flowDsl';

/* ---------- 输出图元 ---------- */

export interface LRect {
  x: number;
  y: number;
  w: number;
  h: number;
  rx: number;
  fill: string;
  stroke: string;
}

export interface LAccent {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
}

export interface LText {
  x: number;
  y: number;
  text: string;
  font: string;
  fill: string;
  italic?: boolean;
  anchor?: 'start' | 'middle';
}

/** arrow 为真时末端画箭头 —— 只用在连线的最后一段上 */
export interface LPath {
  d: string;
  stroke: string;
  arrow?: boolean;
}

export interface LLayout {
  width: number;
  height: number;
  rects: LRect[];
  accents: LAccent[];
  texts: LText[];
  paths: LPath[];
}

/* ---------- 度量与排版常量 ---------- */

const PAD = 10;
const ACCENT_W = 3;
const TEXT_LEFT = PAD + ACCENT_W + 5;
const TEXT_RIGHT = PAD + 4;
const GAP_X = 8;
const GAP_Y = 26;
const SECTION_GAP = 20;
const NOTE_INDENT = 22;
const MIN_BOX_W = 142;

export const FONT_TITLE = '600 12px system-ui, "Segoe UI", Roboto, sans-serif';
export const FONT_LABEL = '600 13px system-ui, "Segoe UI", Roboto, sans-serif';
export const FONT_SHAPE = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
export const FONT_DESC = '11px system-ui, "Segoe UI", Roboto, sans-serif';

const LH_TITLE = 22;
const LH_LABEL = 18;
const LH_SHAPE_EXTRA = 16;
const LH_DESC = 15;
const LH_NOTE = 16;

export const TONE_FILL: Record<Tone, { fill: string; stroke: string; text: string }> = {
  keep: { fill: 'rgba(16, 185, 129, 0.12)', stroke: 'rgba(16, 185, 129, 0.35)', text: '#10b981' },
  drop: { fill: 'rgba(239, 68, 68, 0.1)', stroke: 'rgba(239, 68, 68, 0.35)', text: '#ef4444' },
  info: { fill: 'rgba(59, 130, 246, 0.1)', stroke: 'rgba(59, 130, 246, 0.35)', text: 'var(--accent-color)' },
};

export const COLOR = {
  boxFill: 'var(--bg-primary)',
  boxStroke: 'var(--border-color)',
  bandFill: 'var(--bg-secondary)',
  label: 'var(--text-primary)',
  shape: '#10b981',
  desc: 'var(--text-tertiary)',
  line: 'var(--text-tertiary)',
  start: 'var(--accent-color)',
  end: '#10b981',
  mid: 'var(--border-color)',
};

/* ---------- 文本度量 ---------- */

/** CJK 表意文字 + 中文标点 + 全角符号：这些字符可逐字断行，拉丁词则整体移动 */
const CJK = /[⺀-鿿　-〿＀-￯]/;

let metrics: CanvasRenderingContext2D | null | undefined;

function fontPx(font: string): number {
  const match = font.match(/(\d+(?:\.\d+)?)px/);
  return match ? Number(match[1]) : 12;
}

/**
 * 用 canvas 的字体度量量文本宽度 —— 与浏览器实际排版同一个引擎，比按字符数估算准得多。
 * 校验脚本跑在 Node 上没有 document，退化成保守估算（CJK 按 1em、其余按 0.55em）。
 */
function measure(text: string, font: string): number {
  if (metrics === undefined) {
    metrics = typeof document !== 'undefined' ? document.createElement('canvas').getContext('2d') : null;
  }
  if (metrics) {
    metrics.font = font;
    return metrics.measureText(text).width;
  }
  const size = fontPx(font);
  let width = 0;
  for (const ch of text) width += CJK.test(ch) ? size : size * 0.55;
  return width;
}

/** 切块：CJK 与全角标点单字成块（可任意断行），拉丁单词连同其后的空格成块 */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  let buffer = '';
  for (const ch of text) {
    if (CJK.test(ch)) {
      if (buffer) tokens.push(buffer);
      buffer = '';
      tokens.push(ch);
    } else if (ch === ' ') {
      tokens.push(buffer + ch);
      buffer = '';
    } else {
      buffer += ch;
    }
  }
  if (buffer) tokens.push(buffer);
  return tokens;
}

/** 贪心折行；单个块本身就超宽时（例如很长的 shape 串）按字符硬断，保证不溢出盒子 */
export function wrap(text: string, font: string, maxWidth: number): string[] {
  const lines: string[] = [];
  let line = '';
  const flush = () => {
    if (line !== '') lines.push(line.trimEnd());
    line = '';
  };

  for (const token of tokenize(text)) {
    if (line !== '' && measure(line + token, font) > maxWidth) flush();
    if (measure(token, font) <= maxWidth) {
      line += token;
      continue;
    }
    for (const ch of token) {
      if (line !== '' && measure(line + ch, font) > maxWidth) flush();
      line += ch;
    }
  }
  flush();
  return lines.length > 0 ? lines : [''];
}

/* ---------- 布局 ---------- */

interface Cell {
  /** 盒子里从上到下的文本行，已折行 */
  rows: LText[];
  height: number;
}

function buildSegmentCells(
  segment: FlowSegment,
  boxWidth: number,
  boxX: number,
  boxTop: number,
): Cell {
  const maxWidth = boxWidth - TEXT_LEFT - TEXT_RIGHT;
  const rows: LText[] = [];
  let cursor = boxTop + PAD;

  const labelLines = wrap(segment.label, FONT_LABEL, maxWidth);
  for (const text of labelLines) {
    cursor += LH_LABEL;
    rows.push({ x: boxX + TEXT_LEFT, y: cursor - 5, text, font: FONT_LABEL, fill: COLOR.label });
  }

  if (segment.shape) {
    const shapeLines = wrap(segment.shape, FONT_SHAPE, maxWidth);
    for (const text of shapeLines) {
      cursor += LH_SHAPE_EXTRA;
      rows.push({ x: boxX + TEXT_LEFT, y: cursor - 4, text, font: FONT_SHAPE, fill: COLOR.shape });
    }
  }

  if (segment.desc) {
    const descLines = wrap(segment.desc, FONT_DESC, maxWidth);
    for (const text of descLines) {
      cursor += LH_DESC;
      rows.push({ x: boxX + TEXT_LEFT, y: cursor - 4, text, font: FONT_DESC, fill: COLOR.desc });
    }
  }

  return { rows, height: cursor - boxTop + PAD };
}

/** 把来源点与目标点接起来：各自垂直接到一条横向总线，再垂直落到目标，末端带箭头 */
function connector(sources: number[], y0: number, targets: number[], y1: number, stroke: string): LPath[] {
  const busY = y0 + (y1 - y0) * 0.55;
  const xs = [...sources, ...targets];
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);

  let trunk = sources.map((x) => `M ${x} ${y0} L ${x} ${busY}`).join(' ');
  if (xMax - xMin > 0.5) trunk += ` M ${xMin} ${busY} L ${xMax} ${busY}`;

  const paths: LPath[] = [];
  if (trunk) paths.push({ d: trunk, stroke });
  for (const x of targets) {
    paths.push({ d: `M ${x} ${busY} L ${x} ${y1}`, stroke, arrow: true });
  }
  return paths;
}

export function layoutFlow(doc: FlowDoc, containerWidth: number): LLayout {
  const rects: LRect[] = [];
  const accents: LAccent[] = [];
  const texts: LText[] = [];
  const paths: LPath[] = [];

  // 最宽的那组并行分支决定画布宽度：宁可让外层横向滚动，也不把盒子挤到读不了
  let maxColumns = 1;
  for (const section of doc.sections) {
    for (const block of section.blocks) {
      if (block.kind === 'branches') maxColumns = Math.max(maxColumns, block.items.length);
    }
  }
  const width = Math.max(containerWidth, maxColumns * MIN_BOX_W + (maxColumns - 1) * GAP_X);

  let y = 0;

  for (const section of doc.sections) {
    if (section.title) {
      texts.push({
        x: section.level === 2 ? NOTE_INDENT : 0,
        y: y + 12,
        text: section.title,
        font: FONT_TITLE,
        fill: COLOR.label,
      });
      y += LH_TITLE;
    }

    const flowIndexes = section.blocks
      .map((block, index) =>
        block.kind === 'stage' || block.kind === 'branches' || block.kind === 'loss' || block.kind === 'verdict'
          ? index
          : -1,
      )
      .filter((index) => index >= 0);
    const firstFlow = flowIndexes[0];
    const lastFlow = flowIndexes[flowIndexes.length - 1];
    const tintAt = (index: number) => {
      if (index === firstFlow) return COLOR.start;
      if (index === lastFlow) return COLOR.end;
      return COLOR.mid;
    };

    // 旁注不参与连线，出现时把链条断开 —— 免得画出「箭头指向脚注」
    let previous: { sources: number[]; bottom: number } | null = null;

    for (let index = 0; index < section.blocks.length; index++) {
      const block = section.blocks[index];

      if (block.kind === 'note') {
        previous = null;
        for (const line of wrap(block.text, FONT_DESC, width - NOTE_INDENT)) {
          texts.push({
            x: NOTE_INDENT,
            y: y + 11,
            text: line,
            font: FONT_DESC,
            fill: COLOR.desc,
            italic: true,
          });
          y += LH_NOTE;
        }
        y += 8;
        continue;
      }

      const tint = tintAt(index);

      /* 一行 = 1 个盒子（普通节点）或 N 个并排盒子（并行分支） */
      const segments: FlowSegment[] =
        block.kind === 'branches' ? block.items : block.kind === 'stage' ? [block.segment] : [];
      const isBand = block.kind === 'loss' || block.kind === 'verdict';

      if (isBand) {
        const tone = block.kind === 'verdict' ? TONE_FILL[block.tone] : null;
        const font = block.kind === 'verdict' ? FONT_DESC : FONT_SHAPE;
        const fill = tone ? tone.fill : COLOR.bandFill;
        const stroke = tone ? tone.stroke : COLOR.boxStroke;
        const lines = wrap(block.text, font, width - 2 * (PAD + 4));

        const height = lines.length * LH_DESC + 2 * PAD;
        if (previous) paths.push(...connector(previous.sources, previous.bottom, [width / 2], y, COLOR.line));

        rects.push({ x: 0, y, w: width, h: height, rx: 6, fill, stroke });
        lines.forEach((text, lineIndex) => {
          texts.push({
            x: PAD + 4,
            y: y + PAD + LH_DESC * (lineIndex + 1) - 4,
            text,
            font,
            fill: tone ? tone.text : COLOR.label,
          });
        });

        previous = { sources: [width / 2], bottom: y + height };
        y += height + GAP_Y;
        continue;
      }

      const columns = segments.length;
      const boxWidth = (width - (columns - 1) * GAP_X) / columns;
      const cells = segments.map((segment, columnIndex) =>
        buildSegmentCells(segment, boxWidth, columnIndex * (boxWidth + GAP_X), y),
      );
      const rowHeight = Math.max(...cells.map((cell) => cell.height));
      const centers = cells.map((_, columnIndex) => columnIndex * (boxWidth + GAP_X) + boxWidth / 2);

      if (previous) paths.push(...connector(previous.sources, previous.bottom, centers, y, COLOR.line));

      cells.forEach((cell, columnIndex) => {
        const boxX = columnIndex * (boxWidth + GAP_X);
        rects.push({
          x: boxX,
          y,
          w: boxWidth,
          h: rowHeight,
          rx: 6,
          fill: COLOR.boxFill,
          stroke: COLOR.boxStroke,
        });
        accents.push({ x: boxX + 1, y: y + 5, w: ACCENT_W, h: rowHeight - 10, fill: tint });
        texts.push(...cell.rows);
      });

      previous = { sources: centers, bottom: y + rowHeight };
      y += rowHeight + GAP_Y;
    }

    y += SECTION_GAP - GAP_Y;
  }

  return { width, height: Math.max(y - SECTION_GAP + GAP_Y, 1), rects, accents, texts, paths };
}