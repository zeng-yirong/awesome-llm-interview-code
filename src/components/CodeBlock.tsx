import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface CodeBlockProps {
  code: string;
  language: string;
}

// 高亮规则（按优先级排序：靠前的先匹配）
const RULES: { pattern: RegExp; className: string }[] = [
  { pattern: /#[^\n]*/g, className: 'text-gray-500 italic' },
  { pattern: /"""[\s\S]*?"""/g, className: 'text-gray-500' },
  { pattern: /'[^']*'/g, className: 'text-emerald-400' },
  { pattern: /"[^"]*"/g, className: 'text-emerald-400' },
  { pattern: /@\w+/g, className: 'text-yellow-400' },
  { pattern: /\b(import|from|class|def|return|if|else|elif|for|while|in|not|and|or|is|None|True|False|self|super|with|as|try|except|raise|pass|break|continue|lambda|yield|assert|global|nonlocal)\b/g, className: 'text-purple-400 font-medium' },
  { pattern: /\b(int|float|str|bool|list|dict|tuple|Optional|List|Dict|Tuple|Callable|Any|Union)\b/g, className: 'text-cyan-400' },
  { pattern: /\b\d+\.?\d*\b/g, className: 'text-orange-400' },
  { pattern: /\b(torch|nn|F|math)\b/g, className: 'text-rose-400' },
];

// 合成单遍正则：从左到右扫描，同一位置上靠前的备选优先 —— 与逐条应用规则
// 的优先级一致。关键是「一次扫描」：插入的 span 不会再被后续规则二次匹配，
// 因此不需要占位符。
const COMBINED = new RegExp(
  RULES.map((rule, i) => `(?<r${i}>${rule.pattern.source})`).join('|'),
  'g',
);

export default function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const highlight = (code: string): string => {
    // 第一步：先转义 HTML 特殊字符
    const escaped = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // 第二步：单遍扫描，命中的分支用命名捕获组反查是哪条规则
    return escaped.replace(COMBINED, (match: string, ...rest: unknown[]) => {
      const groups = rest[rest.length - 1] as Record<string, string | undefined>;
      for (let i = 0; i < RULES.length; i++) {
        if (groups[`r${i}`] !== undefined) {
          return `<span class="${RULES[i].className}">${match}</span>`;
        }
      }
      return match;
    });
  };

  return (
    <div className="rounded-lg overflow-hidden border" style={{
      backgroundColor: '#1e293b',
      borderColor: 'var(--border-color)'
    }}>
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 border-b"
        style={{
          backgroundColor: '#0f172a',
          borderColor: 'var(--border-color)'
        }}
      >
        <span className="text-xs font-mono text-gray-400">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors px-2 py-1 rounded hover:bg-gray-700"
        >
          {copied ? (
            <>
              <Check size={14} className="text-green-400" />
              <span className="text-green-400">已复制</span>
            </>
          ) : (
            <>
              <Copy size={14} />
              <span>复制</span>
            </>
          )}
        </button>
      </div>

      {/* Code */}
      <div className="overflow-x-auto p-4">
        <pre className="text-sm leading-relaxed">
          <code
            className="text-gray-300 font-mono"
            dangerouslySetInnerHTML={{ __html: highlight(code) }}
          />
        </pre>
      </div>
    </div>
  );
}
