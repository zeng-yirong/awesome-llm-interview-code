import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface CodeBlockProps {
  code: string;
  language: string;
}

export default function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const highlight = (code: string): string => {
    // 第一步：先转义 HTML 特殊字符
    let escaped = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    // 第二步：使用占位符方法，避免多次替换导致的嵌套问题
    const placeholders: Map<string, string> = new Map();
    let placeholderId = 0;
    
    // 定义高亮规则（按优先级排序）
    const rules = [
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
    
    // 按优先级应用每个规则，使用占位符
    for (const rule of rules) {
      escaped = escaped.replace(rule.pattern, (match) => {
        const placeholder = `\u0000${placeholderId}\u0000`;
        placeholders.set(placeholder, `<span class="${rule.className}">${match}</span>`);
        placeholderId++;
        return placeholder;
      });
    }
    
    // 第三步：恢复占位符
    placeholders.forEach((value, key) => {
      escaped = escaped.replace(key, value);
    });
    
    return escaped;
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
