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
    <div className="rounded-lg overflow-hidden bg-gray-950 border border-gray-800">
      <div className="flex items-center justify-between px-3 py-1.5 bg-gray-900/80 border-b border-gray-800">
        <span className="text-[10px] text-gray-500 font-mono uppercase">{language}</span>
        <button onClick={handleCopy}
          className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-white transition-colors px-1.5 py-0.5 rounded hover:bg-gray-800">
          {copied ? <><Check size={11} className="text-green-400" /><span className="text-green-400">OK</span></>
                  : <><Copy size={11} /><span>Copy</span></>}
        </button>
      </div>
      <div className="overflow-x-auto p-3">
        <pre className="text-[13px] leading-relaxed">
          <code className="text-gray-300 font-mono" dangerouslySetInnerHTML={{ __html: highlight(code) }} />
        </pre>
      </div>
    </div>
  );
}
