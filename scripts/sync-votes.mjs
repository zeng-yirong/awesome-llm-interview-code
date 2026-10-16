#!/usr/bin/env node
/**
 * 把票箱 issue 上的 👍/👎 同步成站点读的 votes.json。
 *
 * 设计要点
 *   - 票箱是 issue：一题一个，带 `vote` 标签，👍 = 热度高 / 👎 = 热度低。
 *     GitHub 账号天然保证「同一个人对同一个 emoji 只能有一个反应」，不用自己防刷。
 *   - 数据写在 `votes-data` 分支，不写 main：main 开了分支保护，bot 推不上去；
 *     而且站点是直接 fetch 那个分支的 raw 文件，票数更新不需要重新构建/部署。
 *   - issue 自动补建：新加了题目，跑一次就给它开票箱，不用手工维护。
 *
 * 需要环境变量
 *   GITHUB_TOKEN        —— Actions 里用 secrets.GITHUB_TOKEN
 *   GITHUB_REPOSITORY   —— Actions 自动注入，形如 owner/repo
 *
 * 用法
 *   node scripts/sync-votes.mjs --dry-run   # 只打印计划，不建 issue、不写分支
 *   node scripts/sync-votes.mjs             # 真的干活
 */
import { Buffer } from 'node:buffer';

const API = 'https://api.github.com';
const GRAPHQL = 'https://api.github.com/graphql';
const DATA_BRANCH = 'votes-data';
const DATA_PATH = 'votes.json';

const LABEL = {
  name: 'vote',
  color: '0e8a16',
  description: '投票票箱：👍 热度高 / 👎 热度低（脚本自动创建，勿删）',
};

/** issue body 里的机器可读标记：靠它把 issue 认回题目，比按标题匹配稳 */
const marker = (id) => `<!-- vote-id: ${id} -->`;
const MARKER_RE = /<!--\s*vote-id:\s*([^\s>]+)\s*-->/;

const dryRun = process.argv.includes('--dry-run');

const token = process.env.GITHUB_TOKEN;
const repoSlug = process.env.GITHUB_REPOSITORY;
if (!token || !repoSlug) {
  console.error('缺少 GITHUB_TOKEN 或 GITHUB_REPOSITORY。本地试跑：');
  console.error('  GITHUB_TOKEN=$(gh auth token) GITHUB_REPOSITORY=zeng-yirong/awesome-llm-interview-code node scripts/sync-votes.mjs --dry-run');
  process.exit(1);
}
const [owner, name] = repoSlug.split('/');

const { problems } = await import(new URL('../src/data/problems.ts', import.meta.url).href);

/* ---------- GitHub API ---------- */

async function api(path, init = {}) {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: {
      accept: 'application/vnd.github+json',
      authorization: `Bearer ${token}`,
      'x-github-api-version': '2022-11-28',
      ...(init.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw new Error(`${init.method ?? 'GET'} ${path} → ${res.status} ${await res.text()}`);
  }
  return res.status === 204 ? null : res.json();
}

async function graphql(query, variables) {
  const res = await fetch(GRAPHQL, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify({ query, variables }),
  });
  const json = await res.json();
  if (json.errors?.length) {
    throw new Error(`GraphQL: ${json.errors.map((e) => e.message).join('; ')}`);
  }
  return json.data;
}

/** 一次请求拿 100 个 issue 的 reaction 计数，比逐个 REST 请求省得多 */
const ISSUES_QUERY = `
query ($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(
      labels: ["vote"]
      states: [OPEN, CLOSED]
      first: 100
      after: $cursor
      orderBy: { field: CREATED_AT, direction: ASC }
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        url
        title
        body
        reactions { THUMBS_UP THUMBS_DOWN }
      }
    }
  }
}`;

async function listVoteIssues() {
  const issues = [];
  let cursor = null;
  for (;;) {
    const data = await graphql(ISSUES_QUERY, { owner, name, cursor });
    const conn = data.repository.issues;
    issues.push(...conn.nodes);
    if (!conn.pageInfo.hasNextPage) return issues;
    cursor = conn.pageInfo.endCursor;
  }
}

async function ensureLabel() {
  try {
    await api(`/repos/${owner}/${name}/labels/${encodeURIComponent(LABEL.name)}`);
  } catch {
    if (dryRun) {
      console.log(`  [dry-run] 会创建标签 ${LABEL.name}`);
      return;
    }
    await api(`/repos/${owner}/${name}/labels`, { method: 'POST', body: JSON.stringify(LABEL) });
    console.log(`  ✓ 创建标签 ${LABEL.name}`);
  }
}

function issueBody(problem) {
  return [
    marker(problem.id),
    '',
    `**${problem.titleCn}** (${problem.title}) · \`${problem.id}\``,
    '',
    `${problem.oneLiner}`,
    '',
    '这道题在面试里的热度，用表情表个态：',
    '',
    '- 👍 **热度高** —— 面试真会考，值得花时间手撕',
    '- 👎 **热度低** —— 偏冷门 / 了解原理即可',
    '',
    '票数会同步到站点（每 6 小时一次）：<https://zeng-yirong.github.io/awesome-llm-interview-code/>',
    '',
    '<sub>本 issue 由 `scripts/sync-votes.mjs` 自动创建，**请勿删除**（删了票数就没了）。题目内容有问题请另开「题目纠错」issue。</sub>',
  ].join('\n');
}

async function createIssue(problem) {
  const body = issueBody(problem);
  if (dryRun) {
    console.log(`  [dry-run] 会创建票箱 #? ${problem.titleCn} (${problem.id})`);
    return null;
  }
  const created = await api(`/repos/${owner}/${name}/issues`, {
    method: 'POST',
    body: JSON.stringify({ title: `[投票] ${problem.titleCn}`, body, labels: [LABEL.name] }),
  });
  console.log(`  ✓ 创建票箱 #${created.number} ${problem.titleCn} (${problem.id})`);
  return created;
}

/* ---------- 写 votes-data 分支 ---------- */

async function writeDataBranch(payload) {
  const content = Buffer.from(`${JSON.stringify(payload, null, 2)}\n`, 'utf8').toString('base64');

  // 分支不存在就从默认分支的头建一个
  try {
    await api(`/repos/${owner}/${name}/git/ref/heads/${DATA_BRANCH}`);
  } catch {
    const repo = await api(`/repos/${owner}/${name}`);
    const base = await api(`/repos/${owner}/${name}/git/ref/heads/${repo.default_branch}`);
    await api(`/repos/${owner}/${name}/git/refs`, {
      method: 'POST',
      body: JSON.stringify({ ref: `refs/heads/${DATA_BRANCH}`, sha: base.object.sha }),
    });
    console.log(`  ✓ 创建分支 ${DATA_BRANCH}`);
  }

  // 文件已存在时必须带上 sha，否则 409
  let fileSha;
  try {
    const existing = await api(`/repos/${owner}/${name}/contents/${DATA_PATH}?ref=${DATA_BRANCH}`);
    fileSha = existing.sha;
  } catch {
    // 首次写入，没有 sha 也正常
  }

  const res = await api(`/repos/${owner}/${name}/contents/${DATA_PATH}`, {
    method: 'PUT',
    body: JSON.stringify({
      message: `chore: 同步投票数据 ${payload.updatedAt}`,
      content,
      branch: DATA_BRANCH,
      ...(fileSha ? { sha: fileSha } : {}),
    }),
  });

  return res.commit?.html_url ?? null;
}

/* ---------- 主流程 ---------- */

console.log(`${dryRun ? '[dry-run] ' : ''}仓库 ${owner}/${name} · 题目 ${problems.length} 道`);

await ensureLabel();

const issues = await listVoteIssues();
console.log(`读到 ${issues.length} 个票箱 issue`);

/** id → issue */
const byId = new Map();
for (const issue of issues) {
  const id = MARKER_RE.exec(issue.body ?? '')?.[1];
  if (id) byId.set(id, issue);
}

// 补建缺失的票箱
const orphanIssueIds = issues
  .map((i) => MARKER_RE.exec(i.body ?? '')?.[1])
  .filter((id) => id && !problems.some((p) => p.id === id));
if (orphanIssueIds.length > 0) {
  console.log(`⚠ 有票箱 issue 找不到对应题目（题目被删了？）：${orphanIssueIds.join(', ')}`);
}

let created = 0;
for (const problem of problems) {
  if (byId.has(problem.id)) continue;
  const issue = await createIssue(problem);
  if (issue) {
    byId.set(problem.id, issue);
    created += 1;
  }
}
if (created === 0) console.log('  没有需要新建的票箱');

// 组装数据
const votes = {};
for (const problem of problems) {
  const issue = byId.get(problem.id);
  if (!issue) continue; // dry-run 下刚建的票箱拿不到编号，跳过
  votes[problem.id] = {
    number: issue.number,
    url: issue.url,
    up: issue.reactions?.THUMBS_UP ?? 0,
    down: issue.reactions?.THUMBS_DOWN ?? 0,
  };
}

const payload = {
  updatedAt: new Date().toISOString(),
  source: `https://github.com/${owner}/${name}`,
  votes,
};

/* 打印社区票 vs 作者标定的 🔥 —— 「只展示」的前提下，这份对照是给你手工修 hot 用的 */
const rows = problems
  .map((p) => ({ p, v: votes[p.id] }))
  .filter((r) => r.v)
  .map((r) => {
    const total = r.v.up + r.v.down;
    return { ...r, total, upRatio: total > 0 ? r.v.up / total : 0 };
  })
  .sort((a, b) => b.upRatio - a.upRatio || b.total - a.total);

console.log('\n社区票 vs 作者标定的热度（按 👍 占比排序）：');
console.log('  👍占比  票数  标定  题目');
for (const r of rows) {
  const ratio = r.total > 0 ? `${Math.round(r.upRatio * 100)}%` : '—';
  const flag = r.total >= 5 && r.p.hot === 3 && r.upRatio < 0.5 ? '  ← 标了必考但社区不认' : '';
  console.log(
    `  ${ratio.padStart(5)}  ${String(r.total).padStart(4)}  ${'🔥'.repeat(r.p.hot).padEnd(4)}  ${r.p.titleCn}${flag}`,
  );
}

if (dryRun) {
  console.log('\n[dry-run] 到此为止，没有创建 issue、没有写分支。');
  process.exit(0);
}

const commitUrl = await writeDataBranch(payload);
console.log(`\n✓ 已写入 ${DATA_BRANCH}/${DATA_PATH}（${Object.keys(votes).length} 题）`);
if (commitUrl) console.log(`  ${commitUrl}`);