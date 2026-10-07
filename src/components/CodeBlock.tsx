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
    // 先转义 HTML 特殊字符
    const escapeHtml = (str: string): string => {
      return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    };
    
    // 使用占位符方法避免 HTML 转义和语法高亮互相干扰
    const placeholders: string[] = [];
    let index = 0;
    
    const addPlaceholder = (match: string, className: string): string => {
      const placeholder = `\u0000${index}\u0000`;
      // 对匹配内容进行 HTML 转义
      placeholders.push(`<span class="${className}">${escapeHtml(match)}</span>`);
      index++;
      return placeholder;
    };
    
    // 语法高亮（使用占位符）
    let result = code
      // Comments
      .replace(/(#.*)$/gm, (match) => addPlaceholder(match, 'text-gray-500 italic'))
      .replace(/("""[\s\S]*?""")/g, (match) => addPlaceholder(match, 'text-gray-500'))
      // Strings
      .replace(/('[^']*')/g, (match) => addPlaceholder(match, 'text-emerald-400'))
      .replace(/("[^"]*")/g, (match) => addPlaceholder(match, 'text-emerald-400'))
      // Keywords
      .replace(/\b(import|from|class|def|return|if|else|elif|for|while|in|not|and|or|is|None|True|False|self|super|with|as|try|except|raise|pass|break|continue|lambda|yield|assert|global|nonlocal)\b/g,
        (match) => addPlaceholder(match, 'text-purple-400 font-medium'))
      // Types
      .replace(/\b(int|float|str|bool|list|dict|tuple|Optional|List|Dict|Tuple|Callable|Any|Union)\b/g,
        (match) => addPlaceholder(match, 'text-cyan-400'))
      // Decorators
      .replace(/(@\w+)/g, (match) => addPlaceholder(match, 'text-yellow-400'))
      // Numbers
      .replace(/\b(\d+\.?\d*)\b/g, (match) => addPlaceholder(match, 'text-orange-400'))
      // torch/nn/F
      .replace(/\b(torch|nn|F|math)\b/g, (match) => addPlaceholder(match, 'text-rose-400'));
    
    // 转义剩余的 HTML 特殊字符（不在占位符中的部分）
    result = result
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    // 恢复占位符为 HTML 标签
    result = result.replace(/\u0000(\d+)\u0000/g, (_, i) => placeholders[parseInt(i)]);
    
    return result;
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
