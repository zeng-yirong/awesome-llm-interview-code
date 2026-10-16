import { useEffect, useState } from 'react';

/**
 * 社区投票数据：一题一个 GitHub issue 当票箱，👍 = 热度高，👎 = 热度低。
 *
 * 数据流：
 *   issue reaction  →  .github/workflows/sync-votes.yml  →  votes-data 分支的 votes.json  →  站点
 *
 * 站点优先读 votes-data 分支的 raw 文件：那个分支不进 Pages 构建，所以票数更新
 * 不需要重新部署。拿不到就退回构建时打包进来的 ./votes.json（本地 dev / 离线 / CDN 抽风）。
 *
 * 没有后端：写 reaction 需要认证，静态站点拿不到 token，所以「投票」是把用户送到
 * issue 上点表情。换来的是 GitHub 账号级别的「一人一票」，不用自己防刷。
 */

const REPO = 'zeng-yirong/awesome-llm-interview-code';
const DATA_BRANCH = 'votes-data';
const RAW_URL = `https://raw.githubusercontent.com/${REPO}/${DATA_BRANCH}/votes.json`;

/** 兜底入口：票数没同步下来时，把用户送到所有票箱 issue 的列表 */
export const VOTE_ISSUES_URL = `https://github.com/${REPO}/issues?q=label%3Avote`;

export interface VoteCount {
  /** issue 编号 */
  number: number;
  /** issue 地址，点 Vote Now 就跳这里 */
  url: string;
  up: number;
  down: number;
}

export interface VotesData {
  updatedAt: string | null;
  votes: Record<string, VoteCount>;
}

async function fetchJson(url: string): Promise<VotesData | null> {
  // 手搓超时：tsconfig 的 lib 停在 ES2020，没有 AbortSignal.timeout
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  try {
    const res = await fetch(url, { cache: 'no-cache', signal: controller.signal });
    if (!res.ok) return null;
    const data = (await res.json()) as VotesData;
    return data && typeof data.votes === 'object' ? data : null;
  } catch {
    // 离线 / 拿不到：静默降级，不打扰读者
    return null;
  } finally {
    clearTimeout(timer);
  }
}

let cached: Promise<VotesData | null> | null = null;

/** 全站只拉一次：36 张卡片共用一个请求 */
export function loadVotes(): Promise<VotesData | null> {
  if (!cached) {
    cached = (async () => {
      return (await fetchJson(RAW_URL)) ?? (await fetchJson('./votes.json'));
    })();
  }
  return cached;
}

export function useVotes(): VotesData | null {
  const [data, setData] = useState<VotesData | null>(null);

  useEffect(() => {
    let alive = true;
    loadVotes().then((d) => {
      if (alive) setData(d);
    });
    return () => {
      alive = false;
    };
  }, []);

  return data;
}

/*
 * 本机投票记录。
 *
 * 只能记「点过 Vote Now」这个动作——真正的票在 GitHub 上，站点看不到是谁投的、
 * 也看不到用户后来有没有取消。所以 UI 上的措辞按实情写：这是本浏览器的记录。
 */
const LS_PREFIX = 'llm-vote:';

export function readLocalVote(problemId: string): boolean {
  try {
    return localStorage.getItem(LS_PREFIX + problemId) !== null;
  } catch {
    return false;
  }
}

export function writeLocalVote(problemId: string): void {
  try {
    localStorage.setItem(LS_PREFIX + problemId, new Date().toISOString());
  } catch {
    // 隐私模式 / 存储被禁：无所谓，投票本身照常
  }
}