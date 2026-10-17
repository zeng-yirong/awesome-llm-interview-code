import { useState } from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';
import { VoteCount, VOTE_ISSUES_URL, readLocalVote, writeLocalVote } from '../lib/votes';

interface VoteButtonProps {
  problemId: string;
  vote?: VoteCount;
}

/**
 * 「Vote Now」：把用户送到该题的投票 issue 上点 👍 / 👎。
 *
 * 票数 = GitHub reaction 计数，由定时 workflow 同步，所以点完不会立刻变——
 * tooltip 里写清了节奏，别让用户以为按钮坏了。
 */
export default function VoteButton({ problemId, vote }: VoteButtonProps) {
  const [voted, setVoted] = useState(() => readLocalVote(problemId));

  const up = vote?.up ?? 0;
  const down = vote?.down ?? 0;
  const total = up + down;
  const upRatio = total > 0 ? (up / total) * 100 : 50;

  return (
    // 卡片标题区整体可点（展开/收起），这里必须掐断冒泡，否则点投票会把卡片一起折叠
    <div onClick={(e) => e.stopPropagation()}>
      <a
        href={vote?.url ?? VOTE_ISSUES_URL}
        target="_blank"
        rel="noopener noreferrer"
        onClick={() => {
          writeLocalVote(problemId);
          setVoted(true);
        }}
        title={
          voted
            ? '本浏览器记录：你去投过票（GitHub 上可以随时取消）'
            : '在 GitHub 上投票：👍 热度高（面试真会考）/ 👎 热度低（了解即可）。票数每 6 小时同步一次。'
        }
        className="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-full font-medium transition-all hover:scale-105"
        style={{
          backgroundColor: 'var(--bg-tertiary)',
          border: '1px solid var(--border-color)',
          color: 'var(--text-secondary)',
        }}
      >
        {vote && (
          <>
            <ThumbsUp size={12} style={{ color: '#10b981' }} />
            <span style={{ color: '#10b981' }}>{up}</span>
            <ThumbsDown size={12} style={{ color: '#ef4444' }} />
            <span style={{ color: '#ef4444' }}>{down}</span>
            {/* 高/低占比条：一眼看出社区觉得这题热不热 */}
            <span
              className="inline-flex w-8 h-1.5 rounded-full overflow-hidden"
              style={{ backgroundColor: '#ef4444' }}
              aria-hidden="true"
            >
              <span style={{ width: `${upRatio}%`, backgroundColor: '#10b981' }} />
            </span>
          </>
        )}
        <span style={{ color: 'var(--accent-color)' }}>
          {voted ? '已投过票 ✓' : 'Vote Now'}
        </span>
      </a>
    </div>
  );
}