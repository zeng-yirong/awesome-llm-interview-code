/**
 * 流程图 DSL —— `Problem.flowDiagram` 的解析器。
 *
 * 背景：flowDiagram 原本是手工排版的 ASCII art，中文在等宽字体里是双宽字符，
 * 混进框线后会错位（36 张图里 17 张的竖线可测地漂移）；而本仓库没有加载任何
 * webfont，`font-mono` 最终回退到系统 monospace，中文宽度是否恰好是拉丁字符的
 * 2 倍并无保证 —— 靠字符对齐在这套技术栈下本来就不可靠。
 *
 * 改成行式 DSL 后由浏览器负责排版，对齐问题从根上消失；同一份文本还能继续
 * 充当文档 `## 📊 张量流程图` 里的 fence 正文，doc 与 TS 依旧逐字节一致。
 *
 * 语法（一行一个元素，缩进无语义，空行忽略）：
 *   # 标题      分节；`##` 为二级。同一字符串里放多张子图就靠它
 *   + 分支      连续的 `+` 行合成一组并行分支，渲染成响应式卡片网格
 *   > 旁注      不解析 `::`，整行当文本
 *   ! 判定      含 ✓ / 保留 → keep，含 ✗ / 丢弃 → drop，否则 info
 *   $ 公式      损失 / 复杂度行，等宽强调且允许换行
 *   // 注释     忽略
 *   其他        节点行：`label :: shape :: desc`（若与实测不符则以实测为准）
 *
 * 只要出现任何框线字符、首行不是分节标题、或有一行不符合语法，就返回 null；
 * 调用方回退到旧的 <pre> 渲染。因此迁移可以分批进行，网站不会中途损坏。
 */

/** 判定行的色调 */
export type Tone = 'keep' | 'drop' | 'info';

/** 一个节点 / 一条分支的内容 */
export interface FlowSegment {
  label: string;
  shape?: string;
  desc?: string;
}

/** 分节里的一块内容 */
export type FlowBlock =
  | { kind: 'stage'; segment: FlowSegment }
  | { kind: 'branches'; items: FlowSegment[] }
  | { kind: 'note'; text: string }
  | { kind: 'verdict'; text: string; tone: Tone }
  | { kind: 'loss'; text: string };

export interface FlowSection {
  title?: string;
  level: 1 | 2;
  blocks: FlowBlock[];
}

export interface FlowDoc {
  sections: FlowSection[];
}

/** 旧 ASCII art 的框线字符：出现任何一个就不是 DSL */
const BOX_CHARS = /[│├└┘┐┌┤┬┴─╱╲]/;

/** 行首标记符 */
const MARKERS = '#+>!$';

/** 字段分隔符。正文里不得出现（已核实全库零出现） */
const DELIM = '::';

/** `[B, H, S, D]` 这种张量形状；否则第二个字段视为说明文字 */
function looksLikeShape(text: string): boolean {
  return text.startsWith('[') && text.endsWith(']');
}

/** 解析 `label :: shape :: desc`，字段可省 */
function parseSegment(payload: string): FlowSegment | null {
  const fields = payload.split(DELIM).map((field) => field.trim());
  if (fields.length === 0 || fields.length > 3) return null;
  if (fields[0] === '') return null;

  if (fields.length === 1) return { label: fields[0] };

  const second = fields[1];
  if (second === '') return null;
  if (fields.length === 2) {
    return looksLikeShape(second)
      ? { label: fields[0], shape: second }
      : { label: fields[0], desc: second };
  }

  const third = fields[2];
  return third === ''
    ? { label: fields[0], shape: second }
    : { label: fields[0], shape: second, desc: third };
}

function detectTone(text: string): Tone {
  if (text.includes('✓') || text.includes('保留')) return 'keep';
  if (text.includes('✗') || text.includes('丢弃')) return 'drop';
  return 'info';
}

/**
 * 把 flowDiagram 解析成结构化模型；不是 DSL 就返回 null（调用方走 <pre> 兜底）。
 */
export function parseFlow(source: string): FlowDoc | null {
  if (typeof source !== 'string' || source.trim() === '') return null;
  if (BOX_CHARS.test(source)) return null;

  const sections: FlowSection[] = [];
  let current: FlowSection | null = null;
  let branchRun: FlowSegment[] | null = null;
  let isFirstContentLine = true;

  // 连续的 `+` 行在遇到下一个非分支行时合成一组
  const flushBranches = () => {
    if (branchRun && branchRun.length > 0 && current) {
      current.blocks.push({ kind: 'branches', items: branchRun });
    }
    branchRun = null;
  };

  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === '' || line.startsWith('//')) continue;

    const marker = MARKERS.includes(line[0]) ? line[0] : '';
    const payload = line.slice(marker === '' ? 0 : 1).trim();

    // 首行必须是分节标题 —— 这是「是不是 DSL」的哨兵
    if (isFirstContentLine) {
      isFirstContentLine = false;
      if (marker !== '#') return null;
    }

    if (marker === '#') {
      flushBranches();
      if (payload === '') return null;
      const level: 1 | 2 = payload.startsWith('#') ? 2 : 1;
      const title = payload.replace(/^#+/, '').trim();
      if (title === '') return null;
      current = { title, level, blocks: [] };
      sections.push(current);
      continue;
    }

    // 出现第一个分节标题之前不该有内容行
    if (!current) return null;

    if (marker === '>') {
      flushBranches();
      if (payload === '') return null;
      current.blocks.push({ kind: 'note', text: payload });
      continue;
    }

    if (marker === '!') {
      flushBranches();
      if (payload === '') return null;
      current.blocks.push({ kind: 'verdict', text: payload, tone: detectTone(payload) });
      continue;
    }

    if (marker === '$') {
      flushBranches();
      if (payload === '') return null;
      current.blocks.push({ kind: 'loss', text: payload });
      continue;
    }

    const segment = parseSegment(payload);
    if (!segment) return null;

    if (marker === '+') {
      if (!branchRun) branchRun = [];
      branchRun.push(segment);
      continue;
    }

    flushBranches();
    current.blocks.push({ kind: 'stage', segment });
  }

  flushBranches();

  // 空分节说明写法有问题，不静默接受
  if (sections.length === 0) return null;
  if (sections.some((section) => section.blocks.length === 0)) return null;

  return { sections };
}