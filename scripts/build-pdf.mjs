/**
 * 把 docs/ 下的 36 篇题解排版成一本可打印的 PDF。
 *
 *   node scripts/build-pdf.mjs [--out <dir>] [--keep] [--no-pdf]
 *
 * 流程分三步，刻意拆开是为了确定性：
 *   1. 生成 book.html —— markdown → HTML，公式用 KaTeX 在 Node 侧就渲染好，
 *      mermaid 只留占位 div（它需要真 DOM）。
 *   2. 用 Chrome --dump-dom 跑一遍 JS，让 mermaid 把占位 div 换成内联 SVG，
 *      把【渲染完成后的 DOM】存成 book.rendered.html。之所以不用
 *      `--print-to-pdf` 直接打印带 JS 的页面，是因为打印快照和 mermaid
 *      的异步渲染谁先谁后没有保证。
 *   3. 对 book.rendered.html 做 --print-to-pdf —— 此时页面已是纯静态，
 *      不需要等任何 JS。
 *
 * 依赖只用到仓库 node_modules 里已有的 marked / katex / mermaid，
 * 不额外装包。
 */
import { execFileSync } from 'node:child_process';
import { copyFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import katex from 'katex';
import { Marked } from 'marked';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const opt = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};

const OUT = path.resolve(opt('--out', path.join(ROOT, 'dist-pdf')));
const WORK = path.join(OUT, 'work');
const BUILT = path.join(WORK, 'build');

const BOOK_TITLE = opt('--title', 'LLM 算法学习手册');

// 流程图能用的最大显示尺寸（CSS px，96dpi）。A4 竖版正文区约 688x1024，
// 这里留出小节标题、外边距、边框的余量，保证图连标题一起塞得进一页。
// 高上限调小 → 图更容易塞进当前页剩余空间 → 空白更少，但图上文字也变小。
const PAGE_BOX = { w: 660, h: Number(opt('--diagram-h', '900')) };
// 横版整页（--landscape 时才用到）的可用区
const LAND_BOX = { w: 970, h: 600 };

const CHROME =
  opt('--chrome', null) ??
  [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  ].find((p) => existsSync(p));

// ---------------------------------------------------------------- 分类表

const CATS = {
  '01-attention': ['👁️', 'Attention', '注意力机制'],
  '02-normalization': ['📏', 'Normalization', '归一化'],
  '03-position': ['📍', 'Position', '位置编码'],
  '04-ffn': ['🔗', 'FFN', '前馈网络'],
  '05-loss': ['📉', 'Loss', '损失函数'],
  '06-optimizer': ['⚙️', 'Optimizer', '优化器'],
  '07-rl': ['🎮', 'RL', '强化学习'],
  '08-peft': ['🎯', 'PEFT', '参数高效微调'],
  '09-efficient': ['💾', 'Efficient', '高效训练'],
  '10-inference': ['⚡', 'Inference', '推理加速'],
  '11-sampling': ['🎲', 'Sampling', '采样策略'],
  '12-architecture': ['🏗️', 'Architecture', '模型架构'],
  '13-basics': ['📖', 'Basics', '基础'],
};

/** 把 `缩放点积注意力 (Scaled Dot-Product Attention)` 拆成中英两段 */
function splitTitle(raw) {
  const m = raw.match(/^(.*?)\s*[（(]([A-Za-z][^）)]*)[）)]\s*$/);
  return m ? { cn: m[1].trim(), en: m[2].trim() } : { cn: raw.trim(), en: '' };
}

// ---------------------------------------------------------------- 代码高亮

const PY_KW = new Set(
  `False None True and as assert async await break class continue def del elif else except finally
   for from global if import in is lambda nonlocal not or pass raise return try while with yield match case`
    .split(/\s+/)
    .filter(Boolean),
);
const PY_BI = new Set(
  `print len range super isinstance issubclass enumerate zip map filter sum min max abs all any int float
   str bytes list dict tuple set frozenset bool open type round sorted reversed divmod format repr getattr
   setattr hasattr iter next repr slice id hash callable staticmethod classmethod property NotImplemented
   Exception ValueError TypeError RuntimeError`
    .split(/\s+/)
    .filter(Boolean),
);

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// 一次性扫完，顺序很重要：注释和字符串先吃掉，其余才轮到关键字，
// 否则字符串里的 # 或关键字会被误染色。
const PY_RE = new RegExp(
  [
    '(#[^\\n]*)', // 1 注释
    '("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|[rbfuRBFU]{0,2}"(?:\\\\.|[^"\\\\\\n])*"|[rbfuRBFU]{0,2}\'(?:\\\\.|[^\'\\\\\\n])*\')', // 2 字符串
    '\\b(0[xX][0-9a-fA-F_]+|0[bB][01_]+|\\d[\\d_]*\\.?[\\d_]*(?:[eE][+-]?\\d+)?[jJ]?)\\b', // 3 数字
    '([A-Za-z_]\\w*)', // 4 标识符
    '([+\\-*/%=<>!&|^~@]+)', // 5 运算符
    '(\\s+)', // 6 空白
    '([\\s\\S])', // 7 其它
  ].join('|'),
  'g',
);

function highlightPy(src) {
  let out = '';
  for (const m of src.matchAll(PY_RE)) {
    const [, comment, str, num, ident, op, ws, other] = m;
    if (comment) out += `<span class="c">${esc(comment)}</span>`;
    else if (str) out += `<span class="s">${esc(str)}</span>`;
    else if (num) out += `<span class="n">${esc(num)}</span>`;
    else if (ident) {
      const cls = PY_KW.has(ident)
        ? 'k'
        : PY_BI.has(ident)
          ? 'b'
          : ident === 'self'
            ? 'sf'
            : /^[A-Z]/.test(ident)
              ? 'C'
              : '';
      out += cls ? `<span class="${cls}">${esc(ident)}</span>` : esc(ident);
    } else if (op) out += `<span class="o">${esc(op)}</span>`;
    else out += esc(ws ?? other ?? '');
  }
  return out;
}

// ---------------------------------------------------------------- markdown

const slug = (s) =>
  s
    .replace(/[\s·]+/g, '-')
    .replace(/[^\w\u4e00-\u9fff-]/g, '')
    .toLowerCase();

const marked = new Marked({
  gfm: true,
  renderer: {
    code({ text, lang }) {
      const l = (lang || '').trim().toLowerCase();
      if (l === 'math' || l === 'latex' || l === 'tex') {
        try {
          return `<div class="eq">${katex.renderToString(text, {
            displayMode: true,
            throwOnError: false,
            strict: false,
            output: 'html',
          })}</div>\n`;
        } catch (e) {
          return `<pre class="code mathraw">${esc(text)}</pre>\n`;
        }
      }
      if (l === 'mermaid') {
        // 转义一次即可：div 的 textContent 必须还原成原始 mermaid 源码，
        // 而 mermaid 源码里本身带有 &lt; 之类的 HTML 实体，正好各归各位。
        return `<div class="mermaid" data-mermaid>${esc(text)}</div>\n`;
      }
      if (l === 'python' || l === 'py') {
        return `<pre class="code py">${highlightPy(text)}</pre>\n`;
      }
      return `<pre class="code">${esc(text)}</pre>\n`;
    },
    heading({ tokens, depth }) {
      const html = this.parser.parseInline(tokens);
      const plain = tokens.map((t) => t.raw ?? '').join('');
      return `<h${depth} id="${slug(plain)}">${html}</h${depth}>\n`;
    },
    // 36 篇文档里一个真链接都没有，但正文里会出现 expert[i](x) 这种数学写法，
    // 标准 markdown 会把它吃成链接、把 (x) 当 URL 丢掉。这里原样还原成文本。
    link({ tokens, href }) {
      return `[${this.parser.parseInline(tokens)}](${href})`;
    },
  },
});

/** 把每个 `## 小节` 连它的正文包成一个 .sec，用来控制跨页粒度。
 *  isWide：这道题的流程图比竖版正文宽，给它单独一页横版，免得被缩得看不清。 */
function groupSections(html, isWide) {
  return html
    .split(/(?=<h2[ >])/)
    .map((part, i) => {
      if (i === 0 || !part.startsWith('<h2')) return part;
      const title = (part.match(/^<h2[^>]*>([\s\S]*?)<\/h2>/) || [, ''])[1];
      // 代码那段允许拆页，其余尽量整块；流程图太宽时改用横版整页
      const cls = /代码实现/.test(title)
        ? 'sec stream'
        : /张量流程图/.test(title) && isWide
          ? 'sec land'
          : 'sec';
      return `<section class="${cls}">${part}</section>`;
    })
    .join('');
}

/** 从渲染完成的 HTML 里量出每道题流程图的原始宽度（viewBox 宽，mermaid 用户单位） */
function measureDiagrams(html) {
  const widths = new Map();
  for (const d of html.split(/(?=<section class="doc")/).slice(1)) {
    const id = (d.match(/id="([^"]+)"/) || [])[1];
    const m = d.match(/class="flowchart"[^>]*viewBox="([^"]+)"/);
    if (id && m) widths.set(id, Number(m[1].split(/\s+/)[2]));
  }
  return widths;
}

// ---------------------------------------------------------------- 读文档

function loadDocs() {
  const docsDir = path.join(ROOT, 'docs');
  const dirs = readdirSync(docsDir)
    .filter((d) => CATS[d])
    .sort();
  const out = [];
  for (const d of dirs) {
    const [emoji, name, cn] = CATS[d];
    const files = readdirSync(path.join(docsDir, d))
      .filter((f) => f.endsWith('.md'))
      .sort();
    for (const f of files) {
      const rel = path.posix.join('docs', d, f);
      const raw = readFileSync(path.join(docsDir, d, f), 'utf8').replace(/\r\n/g, '\n');

      const lines = raw.split('\n');
      const h1 = (lines[0] || '').replace(/^#\s+/, '').trim();
      // 第二行是 `> 副标题`，单独拿出来做导语，不留在正文里
      const sub = /^>\s?/.test(lines[2] ?? '') ? lines[2].replace(/^>\s?/, '').trim() : '';
      const body = lines
        .slice(1)
        .filter((_, i) => !(i === 1 && sub)) // 只滤掉那一行副标题
        .join('\n');

      const { cn: titleCn, en } = splitTitle(h1);
      out.push({
        id: f.replace(/\.md$/, ''),
        file: rel,
        dir: d,
        emoji,
        cat: name,
        catCn: cn,
        titleCn,
        en,
        sub,
        // 小节分块放在 buildHtml 里做，因为要不要横版得等量完图宽才知道
        html: marked.parse(body),
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------- 页面

const CSS = `
:root{
  --ink:#1a1a1a; --dim:#5b6472; --line:#d8dee7; --accent:#1d4ed8;
  --tint:#f4f7fb; --code-bg:#f8fafc;
}
*{box-sizing:border-box}
html{-webkit-print-color-adjust:exact; print-color-adjust:exact}
body{
  margin:0; color:var(--ink); background:#fff;
  font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","Source Han Sans SC",sans-serif;
  font-size:10.5pt; line-height:1.75; text-align:justify;
}
@page{ size:A4; margin:13mm 14mm 13mm 14mm; }
@page land{ size:A4 landscape; margin:12mm 14mm; }

/* 不强制分页：内容连着排，把每一页填满。题与题之间只留一条细分隔线。 */
.doc{
  margin-top:9mm; padding-top:5mm;
  border-top:1px solid var(--line);
  break-inside:auto;
}

/* ---- 封面（压缩成页首一块，不再单独占一整页）---- */
.cover{ padding-top:14mm; text-align:center; }
.cover .kicker{ font-size:9pt; letter-spacing:.45em; color:var(--dim); margin-bottom:4mm; }
.cover h1{ font-size:23pt; margin:0 0 3.5mm; letter-spacing:.06em; border:0; }
.cover .rule{ width:36mm; height:3px; background:var(--accent); margin:0 auto 3.5mm; }
.cover .tagline{ font-size:10.5pt; color:var(--dim); line-height:1.7; }
.cover .cats{ margin-top:4mm; font-size:9pt; color:var(--dim); line-height:1.8; }
.cover .meta{ margin-top:4mm; font-size:9pt; color:var(--dim); }

/* ---- 目录 ---- */
.toc{ padding-top:9mm; }
.toc h2{ font-size:15pt; border:0; padding:0; margin:0 0 5mm; break-after:avoid; }
.toc .grp{ margin-bottom:3.2mm; break-inside:avoid; }
.toc .grp h3{
  font-size:10.5pt; margin:0 0 .8mm; color:var(--accent);
  border:0; padding:0; background:none; letter-spacing:.02em;
}
.toc .items{
  display:flex; flex-wrap:wrap; gap:0 5mm;
  font-size:9.5pt; line-height:1.55; text-align:left;
}
.toc .items a{ color:var(--ink); text-decoration:none; }
.toc .items span{ color:var(--ink); }
.toc .items .no{ color:var(--dim); font-size:8.5pt; margin-right:1mm; }

/* ---- 题目 ---- */
.doc h1{
  font-size:19pt; line-height:1.35; margin:0 0 1.5mm; letter-spacing:.01em;
  break-after:avoid;
}
.doc .en{ font-size:10.5pt; color:var(--dim); letter-spacing:.02em; margin-bottom:2mm; }
.doc .sub{
  font-size:10pt; color:var(--dim); margin:0 0 6mm; padding-bottom:3mm;
  border-bottom:1px solid var(--line);
}
.doc h2{
  font-size:12.5pt; margin:7mm 0 3mm; padding:1.2mm 3mm; border-radius:2px;
  background:var(--tint); border-left:3px solid var(--accent);
  break-after:avoid;
}
.doc h3{ font-size:11pt; margin:4.5mm 0 1.5mm; color:#24405f; break-after:avoid; }
.doc p{ margin:0 0 2.5mm; }
.doc ul,.doc ol{ margin:0 0 3mm; padding-left:6mm; }
.doc ol{ padding-left:7mm; }
.doc li{ margin-bottom:1.2mm; }
.doc strong{ color:#16324f; }
.doc code{
  font-family:Consolas,"Cascadia Mono","Courier New",monospace;
  font-size:9.2pt; background:#eef2f7; padding:.1em .35em; border-radius:2px;
  color:#0f3f6b;
}

/* ---- 公式 ---- */
.eq{ margin:3.5mm 0; break-inside:avoid; text-align:center; overflow:hidden; }
.eq .katex-display{ margin:0; }
.katex{ font-size:1.02em; }
.mathraw{ text-align:left; }

/* ---- 流程图 ---- */
.mermaid{
  margin:4mm 0; padding:3mm 2mm; border:1px solid var(--line); border-radius:3px;
  break-inside:avoid; text-align:center;
}
.mermaid svg{ max-width:100%; height:auto; }

/* ---- 代码 ---- */
.code{
  margin:3mm 0 4mm; padding:3mm 3.5mm; background:var(--code-bg);
  border:1px solid var(--line); border-radius:3px;
  font-family:Consolas,"Cascadia Mono","Courier New",monospace;
  font-size:8.6pt; line-height:1.5; white-space:pre; overflow-wrap:normal;
  text-align:left; break-inside:auto;
}
.code .k{ color:#a626a4; font-weight:600 }
.code .b{ color:#1d4ed8 }
.code .C{ color:#b45309 }
.code .s{ color:#0a7d3e }
.code .c{ color:#8b95a5; font-style:italic }
.code .n{ color:#b45309 }
.code .o{ color:#5b6472 }
.code .sf{ color:#c2410c; font-style:italic }

/* ---- 章节分块 ----
   小节不再整块锁死，允许跨页，页面才填得满。只对"拆开就没意义"的东西
   加 avoid：公式、流程图，以及标题和它后面第一行内容的关系。 */
.sec{ break-inside:auto; }
.sec.stream{ break-inside:auto; }

/* 横版整页：给比竖版正文还宽的流程图，横过来能多用 50% 宽度；
   宽度不超过横版正文的图，这时就是 1:1 原始尺寸，不会被缩放。 */
.land{
  page:land; break-before:page; break-after:page;
  break-inside:avoid;
  /* 横版正文区高约 186mm，取 150mm 做居中下限，留足余量不会溢出到第二页 */
  display:flex; flex-direction:column; justify-content:center; min-height:150mm;
}
.land .mermaid{ margin:0; padding:6mm 4mm; }

/* ---- 分类书眉 ---- */
.band{
  display:flex; align-items:baseline; gap:2mm; margin:0 0 5mm;
  padding:1.6mm 0 1.6mm; border-bottom:2px solid var(--accent);
  font-size:10pt; color:var(--accent); letter-spacing:.04em; break-after:avoid;
}
.band b{ font-size:12pt; }
`;

function buildHtml(docs, katexCss, wide = new Set()) {
  const n = docs.length;
  const groups = [];
  for (const d of docs) {
    const g = groups[groups.length - 1];
    if (g && g.dir === d.dir) g.items.push(d);
    else groups.push({ dir: d.dir, emoji: d.emoji, cat: d.cat, catCn: d.catCn, items: [d] });
  }

  let idx = 0;
  const sections = groups
    .map((g) =>
      g.items
        .map((d) => {
          idx += 1;
          const band =
            g.items.indexOf(d) === 0
              ? `<div class="band">${g.emoji}<b>${g.catCn}</b><span>${g.cat}</span><span style="margin-left:auto;color:#8b95a5">${g.items.length} 题</span></div>`
              : '';
          return `<section class="doc" id="${d.id}">
${band}
<h1>${esc(d.titleCn)}</h1>
${d.en ? `<div class="en">${esc(d.en)}</div>` : ''}
${d.sub ? `<p class="sub">${marked.parseInline(d.sub)}</p>` : ''}
${groupSections(d.html, wide.has(d.id))}</section>`;
        })
        .join('\n'),
    )
    .join('\n');

  let no = 0;
  const toc = groups
    .map(
      (g) => `<div class="grp"><h3>${g.emoji} ${g.catCn} · ${g.cat}</h3><div class="items">${g.items
        .map((d) => {
          no += 1;
          return `<span><a href="#${d.id}"><span class="no">${String(no).padStart(2, '0')}</span>${esc(d.titleCn)}</a></span>`;
        })
        .join('')}</div></div>`,
    )
    .join('\n');

  const today = new Date().toISOString().slice(0, 10);

  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>${BOOK_TITLE}</title>
<style>${katexCss}</style>
<style>${CSS}</style>
</head>
<body>

<section class="cover">
  <div class="kicker">LLM INTERVIEW</div>
  <h1>${BOOK_TITLE}</h1>
  <div class="rule"></div>
  <div class="tagline">
    ${n} 道高频题 · 原理推导 / 核心公式 / 张量流程图 / 代码实现 / 面试要点
  </div>
  <div class="cats">
    ${groups.map((g) => `${g.emoji} ${g.catCn}`).join(' &nbsp;·&nbsp; ')}
  </div>
  <div class="meta">${groups.length} 个主题 · ${n} 篇题解 · 生成于 ${today}</div>
</section>

<section class="page toc">
  <h2>目录</h2>
  ${toc}
</section>

${sections}

<script src="mermaid.min.js"></script>
<script id="boot">
(async () => {
  const nodes = [...document.querySelectorAll('[data-mermaid]')];
  const PAGE_BOX = ${JSON.stringify(PAGE_BOX)};
  const LAND_BOX = ${JSON.stringify(LAND_BOX)};
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'default',
    fontFamily: 'inherit',
    flowchart: { curve: 'basis', htmlLabels: true, useMaxWidth: true },
  });
  for (let i = 0; i < nodes.length; i++) {
    const el = nodes[i];
    try {
      const { svg } = await mermaid.render('mmd-' + i, el.textContent);
      el.innerHTML = svg;
      // 关键：mermaid 只按容器"宽度"给 max-width，又高又窄的图（DAPO 168x1144、
      // 混合精度 216x1176）会高出整整一页，打印时会被整块裁掉。这里按 viewBox
      // 自己算一个能塞进单页的尺寸，宽高两个方向都约束住。
      const s = el.querySelector('svg');
      const vb = (s.getAttribute('viewBox') || '').split(/[\\s,]+/).map(Number);
      const vw = vb[2], vh = vb[3];
      if (vw > 0 && vh > 0) {
        const box = el.closest('.land') ? LAND_BOX : PAGE_BOX;
        const k = Math.min(1, box.w / vw, box.h / vh);
        s.setAttribute('width', Math.round(vw * k));
        s.setAttribute('height', Math.round(vh * k));
        s.style.maxWidth = '100%';
      }
      el.removeAttribute('data-mermaid');
    } catch (e) {
      el.textContent = 'mermaid 渲染失败: ' + (e && e.message ? e.message : e);
      el.setAttribute('data-mermaid-error', '1');
    }
  }
  document.title = 'RENDERED ' + nodes.length;
})();
</script>
</body>
</html>`;
}

// ---------------------------------------------------------------- chrome

function chrome(args) {
  return execFileSync(CHROME, [...args, '--headless=new', '--disable-gpu', '--no-sandbox'], {
    maxBuffer: 1 << 30,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

const fileUrl = (p) => `file:///${p.replace(/\\/g, '/')}`;

function renderWithChrome(htmlPath) {
  console.log('  Chrome 渲染 mermaid ...');
  const out = chrome([
    `--virtual-time-budget=60000`,
    `--window-size=794,1123`,
    '--dump-dom',
    fileUrl(htmlPath),
  ]);
  const html = Buffer.from(out).toString('utf8');

  const done = html.match(/<title>RENDERED (\d+)<\/title>/);
  const total = done ? Number(done[1]) : -1;

  // 打印阶段不需要任何 JS：脚本留着会在已经换成 SVG 的节点上跑第二遍。
  // 而且必须先剥掉脚本再统计，否则统计到的是脚本自己的源码文本。
  const staticHtml = html
    .replace(/<script id="boot">[\s\S]*?<\/script>/, '')
    .replace(/<script src="mermaid\.min\.js"><\/script>/, '')
    .replace(/<title>[^<]*<\/title>/, `<title>${BOOK_TITLE}</title>`);

  const svgs = (staticHtml.match(/<svg[^>]*class="[^"]*flowchart/g) || []).length;
  const errs = (staticHtml.match(/data-mermaid-error="1"/g) || []).length;
  console.log(`  boot 报告 ${total} 张  实际 svg=${svgs}  失败=${errs}`);
  for (const m of staticHtml.matchAll(/mermaid 渲染失败: ([^<]{0,200})/g)) console.log('   ! ' + m[1]);
  if (total !== svgs) console.log('  !! boot 报告的张数与页面上实际 svg 数对不上');

  const renderedPath = path.join(BUILT, 'book.rendered.html');
  writeFileSync(renderedPath, staticHtml, 'utf8');
  return { renderedPath, svgs, errs, total };
}

function printPdf(htmlPath, pdfPath) {
  console.log('  Chrome 打印 PDF ...');
  chrome([
    '--no-pdf-header-footer',
    '--virtual-time-budget=10000',
    `--print-to-pdf=${pdfPath}`,
    fileUrl(htmlPath),
  ]);
}

// ---------------------------------------------------------------- main

function main() {
  if (!CHROME) throw new Error('找不到 chrome.exe，用 --chrome <路径> 指定');
  const docs = loadDocs();
  console.log(`读到 ${docs.length} 篇题解`);

  mkdirSync(BUILT, { recursive: true });
  mkdirSync(path.join(OUT), { recursive: true });

  // KaTeX 的 CSS 直接内联，字体目录摆到 HTML 旁边（CSS 里的 url(fonts/..) 按文档 base 解析）
  const katexCss = readFileSync(path.join(ROOT, 'node_modules/katex/dist/katex.min.css'), 'utf8');
  cpSync(path.join(ROOT, 'node_modules/katex/dist/fonts'), path.join(BUILT, 'fonts'), {
    recursive: true,
  });
  copyFileSync(
    path.join(ROOT, 'node_modules/mermaid/dist/mermaid.min.js'),
    path.join(BUILT, 'mermaid.min.js'),
  );

  const htmlPath = path.join(BUILT, 'book.html');
  const limit = Number(opt('--landscape-min-width', '1000'));

  // 横版整页默认关掉：它必然要强制分页，和"紧凑、不留空白"冲突。
  // 需要给超宽的图争取版面时再开 --landscape。
  let wide = new Set();
  if (flag('--landscape')) {
    const probe = buildHtml(docs, katexCss);
    writeFileSync(htmlPath, probe, 'utf8');
    console.log(`  [量宽] book.html ${(Buffer.byteLength(probe) / 1024).toFixed(0)} KB`);
    const r0 = renderWithChrome(htmlPath);
    const widths = measureDiagrams(readFileSync(r0.renderedPath, 'utf8'));
    for (const [id, w] of widths) if (w > limit) wide.add(id);
    console.log(
      `  量到 ${widths.size} 张图；超过 ${limit} 的走横版整页（${wide.size} 张）: ${[...wide].join(', ') || '无'}`,
    );
  }

  // 第二遍：按量出来的结果重排，这一遍才是要打印的
  const html = buildHtml(docs, katexCss, wide);
  writeFileSync(htmlPath, html, 'utf8');
  console.log(`  book.html  ${(Buffer.byteLength(html) / 1024).toFixed(0)} KB`);

  const r = renderWithChrome(htmlPath);
  if (r.total !== docs.length) console.log(`  !! mermaid 节点数 ${r.total} ≠ 题目数 ${docs.length}`);

  const pdfPath = path.join(OUT, `${BOOK_TITLE}.pdf`);
  if (!flag('--no-pdf')) {
    printPdf(r.renderedPath, pdfPath);
    console.log(`  PDF -> ${pdfPath}`);
  }
  console.log(`中间产物: ${BUILT}`);
}

main();