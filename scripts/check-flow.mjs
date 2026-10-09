#!/usr/bin/env node
/**
 * 只读校验脚本（无副作用，只打印报告）。
 *
 *   1. flowDiagram 的 DSL 解析普查：哪些题已迁移成结构化 DSL、哪些还走 <pre> 兜底
 *   2. 已迁移条目的语法卫生：不含框线字符、每行 `::` 字段 ≤ 3
 *   3. doc ↔ TS 一致性：张量流程图 fence / oneLiner / 原理首段 / 原理分节 / 面试要点条数
 *
 * 用法: node scripts/check-flow.mjs
 */
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');

const { problems } = await import(new URL('../src/data/problems.ts', import.meta.url).href);
const { parseFlow } = await import(new URL('../src/lib/flowDsl.ts', import.meta.url).href);

const BOX_CHARS = /[│├└┘┐┌┤┬┴─╱╲]/;
const failures = [];
const warnings = [];

/* ---------- id ↔ 文档映射 ---------- */
// 仓库里没有 id→路径的映射（文档按 H1 命名，且有多种写法），所以显式列出。
// problems 数组的排列顺序与 README 总览表的显示顺序并不一致，不能按下标 zip。
const DOC_OF = {
  'attn-sdp': 'docs/01-attention/scaled-dot-product-attention.md',
  'attn-mha': 'docs/01-attention/multi-head-attention.md',
  'attn-causal': 'docs/01-attention/causal-mask.md',
  'attn-gqa': 'docs/01-attention/gqa.md',
  'attn-flash': 'docs/01-attention/flash-attention.md',
  'attn-mla': 'docs/01-attention/mla.md',
  'attn-sparse': 'docs/01-attention/sparse-attention.md',
  'attn-kv-cache': 'docs/01-attention/kv-cache.md',
  'norm-ln': 'docs/02-normalization/layer-norm.md',
  'norm-rms': 'docs/02-normalization/rms-norm.md',
  'pos-rope': 'docs/03-position/rope.md',
  'ffn-std': 'docs/04-ffn/ffn.md',
  'ffn-swiglu': 'docs/04-ffn/swiglu.md',
  'ffn-moe': 'docs/04-ffn/moe.md',
  'loss-ce': 'docs/05-loss/cross-entropy.md',
  'loss-dpo': 'docs/05-loss/dpo.md',
  'loss-ppo': 'docs/05-loss/ppo.md',
  'loss-grpo': 'docs/05-loss/grpo.md',
  'loss-gspo': 'docs/05-loss/gspo.md',
  'loss-dapo': 'docs/05-loss/dapo.md',
  'loss-opd': 'docs/05-loss/opd.md',
  'loss-self-opd': 'docs/05-loss/self-opd.md',
  'opt-adamw': 'docs/06-optimizer/adamw.md',
  'opt-muon': 'docs/06-optimizer/muon.md',
  'peft-lora': 'docs/08-peft/lora.md',
  'sample-strategies': 'docs/11-sampling/sampling.md',
  'eff-mixed': 'docs/09-efficient/mixed-precision.md',
  'eff-grad-ckpt': 'docs/09-efficient/gradient-checkpointing.md',
  'eff-fp8': 'docs/09-efficient/fp8-blockwise.md',
  'infer-paged': 'docs/10-inference/paged-attention.md',
  'infer-spec': 'docs/10-inference/speculative-decoding.md',
  'arch-decoder': 'docs/12-architecture/decoder-only.md',
  'arch-mtp': 'docs/12-architecture/mtp.md',
  'rl-gae': 'docs/07-rl/gae.md',
  'basics-backprop': 'docs/13-basics/backprop.md',
  'basics-activation': 'docs/13-basics/activation.md',
};

function buildDocMap() {
  const map = new Map();
  const usedPaths = new Set();
  for (const problem of problems) {
    const relPath = DOC_OF[problem.id];
    if (!relPath) {
      failures.push(`[${problem.id}] 映射表里没有对应文档，请补 DOC_OF`);
      continue;
    }
    if (usedPaths.has(relPath)) {
      failures.push(`[${problem.id}] 文档路径 ${relPath} 被多个题目共用`);
    }
    usedPaths.add(relPath);
    map.set(problem.id, relPath);
  }
  const unmapped = Object.keys(DOC_OF).filter((id) => !problems.some((p) => p.id === id));
  if (unmapped.length > 0) {
    failures.push(`映射表里有不存在的 id: ${unmapped.join(', ')}`);
  }
  return map;
}

/* ---------- markdown 提取 ---------- */

/** 取 `## <heading>` 到下一个 `## ` 之间的正文 */
function sectionBody(markdown, heading) {
  const start = markdown.indexOf(`## ${heading}`);
  if (start < 0) return null;
  const after = markdown.indexOf('\n', start);
  if (after < 0) return null;
  const next = markdown.indexOf('\n## ', after + 1);
  return markdown.slice(after + 1, next < 0 ? markdown.length : next);
}

/**
 * 只保留汉字 / 字母 / 数字。用来区分「文字真的不一样」和「只是标点风格不同」——
 * 文档是给读者看的版本（会用破折号、分号断句），TS 是纯文本版本，
 * 后者允许存在标点差异，前者不允许存在措辞差异。
 */
function wordsOnly(text) {
  return text.replace(/[^\p{Script=Han}\p{L}\p{N}]/gu, '');
}

/** 文档里可以带 markdown 强调，TS 一律存纯文本 —— 比较前按同一约定剥掉。
 * 沿用仓库既有习惯：TS 是「可渲染的纯文本」，文档是带格式的读者版本。 */
function stripMarkup(text) {
  return text.split('`').join('').split('**').join('').trim();
}

/** 取正文里第一个 ``` fence 的内容（不含围栏本身） */
function firstFence(body) {
  if (body === null) return null;
  const match = body.match(/```[a-zA-Z]*\n([\s\S]*?)\n```/);
  return match ? match[1] : null;
}

/** 原理与思想的第一段正文（跳过标题行与 bullet）
 *  `### 核心概念` 是新版三段式的首段标题：它排在段落前面，跳过即可，不能像
 *  `**小节标题**` 那样直接 break（那样会得到空段落）。 */
function firstParagraph(body) {
  const collected = [];
  for (const line of body.split('\n')) {
    const text = line.trim();
    if (text === '') {
      if (collected.length > 0) break;
      continue;
    }
    if (text.startsWith('### ')) {
      if (collected.length > 0) break; // 已经收完首段，后面是下一节
      continue;
    }
    if (text.startsWith('**') || text.startsWith('- ')) break;
    collected.push(text);
  }
  return stripMarkup(collected.join(' '));
}

/** 文档里的分节。仓库里并存两种写法：
 *  - 旧版：`**小节标题**` + `- ` bullet
 *  - 新版三段式：`### 小节标题` + 段落或 `1. ` 编号列表
 *  两种都解析出来，条目一律剥掉前缀，好与 TS 的 items 逐条比对。 */
function docSections(body) {
  const sections = [];
  let current = null;
  let numbered = false; // 新版分节的条目允许不写列表标记（核心思想就是一整段）
  let leadHeading = false; // 三段式的首个 `### ` 标题
  for (const line of body.split('\n')) {
    const text = line.trim();
    if (text === '') continue;
    if (text.startsWith('### ')) {
      // 三段式的第一个标题（`### 核心概念`）标的就是首段本身：站点把 principle
      // 渲染成不带标题的引导段，所以它不对应任何分节，跳过。
      if (sections.length === 0 && !leadHeading) {
        leadHeading = true;
        current = null;
        numbered = false;
        continue;
      }
      leadHeading = true;
      current = { title: text.slice(4).trim(), items: [] };
      sections.push(current);
      numbered = true;
      continue;
    }
    if (text.startsWith('**') && text.endsWith('**') && text.length > 4) {
      current = { title: text.slice(2, -2), items: [] };
      sections.push(current);
      numbered = false;
      continue;
    }
    if (text.startsWith('- ')) {
      if (current) current.items.push(stripMarkup(text.slice(2)));
      continue;
    }
    // 旧版分节里出现无标记的行一律忽略（保持既有行为），新版按条目收下
    if (current && numbered) current.items.push(stripMarkup(text.replace(/^\d+\.\s+/, '')));
  }
  return sections;
}

function countBullets(body) {
  return body.split('\n').filter((line) => line.trim().startsWith('- ')).length;
}

function stripBackticks(text) {
  return text.split('`').join('');
}

/* ---------- 主流程 ---------- */

const docMap = buildDocMap();
const migrated = [];
const legacy = [];

for (const problem of problems) {
  const isDsl = parseFlow(problem.flowDiagram) !== null;
  (isDsl ? migrated : legacy).push(problem.id);

  if (isDsl) {
    if (BOX_CHARS.test(problem.flowDiagram)) {
      failures.push(`[${problem.id}] 已迁移为 DSL，但仍含框线字符`);
    }
    for (const line of problem.flowDiagram.split('\n')) {
      const text = line.trim();
      if (text === '' || text.startsWith('//')) continue;
      const body = text.slice(1).trim();
      const fields = body.split('::').length;
      if (fields > 3) {
        failures.push(`[${problem.id}] 一行有 ${fields} 个 :: 字段（上限 3）: ${text}`);
      }
    }
  }

  const relPath = docMap.get(problem.id);
  if (!relPath) {
    failures.push(`[${problem.id}] 找不到对应文档（README 表映射缺失）`);
    continue;
  }

  const markdown = readFileSync(join(ROOT, relPath), 'utf8').replace(/\r\n/g, '\n');

  // 1) 张量流程图 fence 与 TS 逐字节一致
  const flowBody = sectionBody(markdown, '📊 张量流程图');
  const fence = firstFence(flowBody);
  if (fence === null) {
    failures.push(`[${problem.id}] 文档缺少「张量流程图」fence (${relPath})`);
  } else if (fence !== problem.flowDiagram) {
    const sameWhenTrimmed =
      fence.split('\n').map((l) => l.trimEnd()).join('\n') ===
      problem.flowDiagram.split('\n').map((l) => l.trimEnd()).join('\n');
    failures.push(
      `[${problem.id}] 张量流程图与 TS flowDiagram 不一致（${sameWhenTrimmed ? '仅行尾空白差异' : '内容差异'}）`,
    );
  }

  // 2) oneLiner 与引文一致（文档里的反引号会被剥掉）
  const quoteMatch = markdown.match(/^>\s*(.+)$/m);
  if (!quoteMatch) {
    failures.push(`[${problem.id}] 文档缺少引文行`);
  } else if (stripBackticks(quoteMatch[1].trim()) !== problem.oneLiner) {
    const docQuote = stripBackticks(quoteMatch[1].trim());
    if (wordsOnly(docQuote) === wordsOnly(problem.oneLiner)) {
      warnings.push(`[${problem.id}] oneLiner 与文档引文仅标点风格不同`);
    } else {
      failures.push(`[${problem.id}] oneLiner 与文档引文不一致（措辞不同）`);
    }
  }

  // 3) 原理与思想
  const principleBody = sectionBody(markdown, '📌 原理与思想');
  if (principleBody === null) {
    failures.push(`[${problem.id}] 文档缺少「原理与思想」章节`);
  } else {
    const paragraph = firstParagraph(principleBody);
    if (paragraph !== problem.principle) {
      if (wordsOnly(paragraph) === wordsOnly(problem.principle)) {
        warnings.push(`[${problem.id}] principle 与文档原理首段仅标点风格不同`);
      } else {
        failures.push(`[${problem.id}] principle 与文档原理首段不一致（措辞不同）`);
      }
    }

    const sections = docSections(principleBody);
    const tsSections = problem.principleSections ?? [];
    if (tsSections.length === 0) {
      if (sections.length > 0) {
        warnings.push(`[${problem.id}] 文档已有 ${sections.length} 个分节，但 TS 还没有 principleSections`);
      }
    } else {
      if (sections.length !== tsSections.length) {
        failures.push(
          `[${problem.id}] 原理分节数不一致：文档 ${sections.length} vs TS ${tsSections.length}`,
        );
      }
      const count = Math.min(sections.length, tsSections.length);
      for (let i = 0; i < count; i++) {
        if (sections[i].title !== tsSections[i].title) {
          failures.push(
            `[${problem.id}] 第 ${i + 1} 节标题不一致：文档「${sections[i].title}」 vs TS「${tsSections[i].title}」`,
          );
        }
        if (sections[i].items.join('\n') !== tsSections[i].items.join('\n')) {
          failures.push(`[${problem.id}] 第 ${i + 1} 节（${sections[i].title}）的条目不一致`);
        }
      }
    }
  }

  // 4) 面试要点：报告 doc/TS 的条数漂移。
  //    注意这不是仓库的硬性约定 —— 存量题目里本来就有 doc ≠ TS + 1 的情况，
  //    所以只作为警告输出，本次任务不改动 keyPoints。
  const keyBody = sectionBody(markdown, '🎯 面试要点');
  if (keyBody === null) {
    failures.push(`[${problem.id}] 文档缺少「面试要点」章节`);
  } else {
    const docBullets = countBullets(keyBody);
    if (docBullets !== problem.keyPoints.length + 1) {
      warnings.push(
        `[${problem.id}] 面试要点条数漂移：文档 ${docBullets} vs TS ${problem.keyPoints.length} + 1`,
      );
    }
  }
}

/* ---------- 报告 ---------- */

console.log(`题库总数: ${problems.length}`);
console.log(`已迁移为 DSL: ${migrated.length}  → ${migrated.join(', ') || '(无)'}`);
console.log(`仍走 <pre> 兜底: ${legacy.length}  → ${legacy.join(', ') || '(无)'}`);

const withSections = problems.filter((p) => (p.principleSections ?? []).length > 0);
console.log(`已补 principleSections: ${withSections.length} / ${problems.length}`);

if (warnings.length > 0) {
  console.log(`\n警告 ${warnings.length} 条:`);
  for (const warning of warnings) console.log(`  ! ${warning}`);
}

if (failures.length > 0) {
  console.log(`\n失败 ${failures.length} 条:`);
  for (const failure of failures) console.log(`  ✗ ${failure}`);
  process.exitCode = 1;
} else {
  console.log('\n全部检查通过 ✓');
}