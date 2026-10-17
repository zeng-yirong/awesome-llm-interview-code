#!/usr/bin/env node
/**
 * 把题库和文档里的 Python 片段抽出来，交给 scripts/code-lint.py 做静态体检。
 *
 * 为什么要单独一个脚本：`npm run check` 只管「文档 ↔ TS 是否一致」，
 * 不管代码本身对不对 —— 之前好几个"照抄必崩"的片段（未定义的变量、不存在的
 * self 方法）都是从这条缝里漏出去的。
 *
 * 片段会落到临时目录里做成一个个 .py 再交给 python（用完即删，不污染工作区）。
 * 只依赖 Python 标准库，不需要 torch，也不需要联网。
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DOCS_DIR = fileURLToPath(new URL('../docs/', import.meta.url));
const LINTER = fileURLToPath(new URL('./code-lint.py', import.meta.url));

const { problems } = await import(new URL('../src/data/problems.ts', import.meta.url).href);

/* ---------- 收集片段 ---------- */

const snippets = [];

// 站点版的 code 字段
for (const problem of problems) {
  snippets.push({ label: `${problem.id} · problems.ts`, code: problem.code });
}

// 文档里的每个 ```python 围栏
for (const category of readdirSync(DOCS_DIR)) {
  for (const file of readdirSync(join(DOCS_DIR, category))) {
    if (!file.endsWith('.md')) continue;
    const md = readFileSync(join(DOCS_DIR, category, file), 'utf8');
    let index = 0;
    for (const match of md.matchAll(/```python\r?\n([\s\S]*?)```/g)) {
      index += 1;
      snippets.push({ label: `docs/${category}/${file} · 第 ${index} 个 python 块`, code: match[1] });
    }
  }
}

/* ---------- 落盘 → 交给 python ---------- */

const workDir = mkdtempSync(join(tmpdir(), 'llm-code-lint-'));
const labels = new Map();
snippets.forEach((sn, i) => {
  const name = `f${String(i).padStart(3, '0')}.py`;
  writeFileSync(join(workDir, name), sn.code, 'utf8');
  labels.set(name, sn.label);
});

function findPython() {
  for (const cmd of ['python3', 'python']) {
    try {
      execFileSync(cmd, ['--version'], { stdio: 'ignore' });
      return cmd;
    } catch {
      // 换下一个候选
    }
  }
  return null;
}

const python = findPython();
if (!python) {
  rmSync(workDir, { recursive: true, force: true });
  console.error('找不到 python3 / python。这个体检脚本只用标准库，装个 Python 即可（不需要 torch）。');
  process.exit(1);
}

let raw = '';
try {
  raw = execFileSync(python, [LINTER, workDir], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    // Windows 管道默认走 GBK，中文说明到这边会变乱码
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
} catch (err) {
  rmSync(workDir, { recursive: true, force: true });
  console.error('体检脚本自己挂了：');
  console.error(err.stdout || err.message);
  process.exit(2);
}
rmSync(workDir, { recursive: true, force: true });

/* ---------- 报告 ---------- */

const findings = new Map(); // 可读标签 → [{ line, message }]
for (const line of raw.split('\n')) {
  const match = /^(f\d+\.py):(\d+): (.*)$/.exec(line.trim());
  if (!match) continue;
  const label = labels.get(match[1]) ?? match[1];
  if (!findings.has(label)) findings.set(label, []);
  findings.get(label).push({ line: Number(match[2]), message: match[3] });
}

console.log(
  `Python 片段体检：题库 ${problems.length} 段 + 文档 ${snippets.length - problems.length} 段`,
);

if (findings.size === 0) {
  console.log('全部通过 ✓');
  process.exit(0);
}

let total = 0;
for (const [label, items] of findings) {
  console.log(`\n✗ ${label}`);
  for (const item of items) {
    total += 1;
    console.log(`    第 ${item.line} 行: ${item.message}`);
  }
}
console.log(`\n共 ${total} 处，分布在 ${findings.size} 个片段。`);
console.log('确认是有意引用的外部符号后，在片段里加一行 "# lint:allow 名字1 名字2"；');
console.log('跨片段通用的外部符号加进 scripts/code-lint.py 的 ALLOW。');
process.exit(1);