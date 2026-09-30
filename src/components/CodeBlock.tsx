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
    return code
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      // Comments
      .replace(/(#.*)$/gm, '<span class="text-gray-500 italic">$1</span>')
      .replace(/("""[\s\S]*?""")/g, '<span class="text-gray-500">$1</span>')
      // Strings
      .replace(/('[^']*')/g, '<span class="text-emerald-400">$1</span>')
      .replace(/("[^"]*")/g, '<span class="text-emerald-400">$1</span>')
      // Keywords
      .replace(/\b(import|from|class|def|return|if|else|elif|for|while|in|not|and|or|is|None|True|False|self|super|with|as|try|except|raise|pass|break|continue|lambda|yield|assert|global|nonlocal)\b/g,
        '<span class="text-purple-400 font-medium">$1</span>')
      // Types
      .replace(/\b(int|float|str|bool|list|dict|tuple|Optional|List|Dict|Tuple|Callable|Any|Union)\b/g,
        '<span class="text-cyan-400">$1</span>')
      // Decorators
      .replace(/(@\w+)/g, '<span class="text-yellow-400">$1</span>')
      // Numbers
      .replace(/\b(\d+\.?\d*)\b/g, '<span class="text-orange-400">$1</span>')
      // torch/nn/F
      .replace(/\b(torch|nn|F|math)\b/g, '<span class="text-rose-400">$1</span>');
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
