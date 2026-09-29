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

  // Simple syntax highlighting
  const highlightCode = (code: string): string => {
    let highlighted = code
      // Escape HTML
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      // Comments
      .replace(/(#.*)$/gm, '<span class="text-gray-500 italic">$1</span>')
      .replace(/("""[\s\S]*?""")/g, '<span class="text-gray-500 italic">$1</span>')
      // Strings
      .replace(/('[^']*')/g, '<span class="text-emerald-400">$1</span>')
      .replace(/("[^"]*")/g, '<span class="text-emerald-400">$1</span>')
      // Keywords
      .replace(/\b(import|from|class|def|return|if|else|elif|for|while|in|not|and|or|is|None|True|False|self|super|with|as|try|except|raise|pass|break|continue|lambda|yield|assert|global|nonlocal)\b/g, '<span class="text-purple-400 font-semibold">$1</span>')
      // Types and builtins
      .replace(/\b(int|float|str|bool|list|dict|tuple|set|Optional|List|Dict|Tuple|Callable|Any|Union)\b/g, '<span class="text-cyan-400">$1</span>')
      // Decorators
      .replace(/(@\w+)/g, '<span class="text-yellow-400">$1</span>')
      // Numbers
      .replace(/\b(\d+\.?\d*)\b/g, '<span class="text-orange-400">$1</span>')
      // Function definitions
      .replace(/\b(\w+)\s*\(/g, '<span class="text-blue-400">$1</span>(')
      // torch, nn, F etc
      .replace(/\b(torch|nn|F|math|tl|triton|dist)\b/g, '<span class="text-rose-400">$1</span>');

    return highlighted;
  };

  return (
    <div className="relative group rounded-lg overflow-hidden bg-gray-950 border border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-800">
        <span className="text-xs text-gray-400 font-mono">{language}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors px-2 py-1 rounded hover:bg-gray-800"
        >
          {copied ? (
            <>
              <Check size={14} className="text-green-400" />
              <span className="text-green-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy size={14} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      {/* Code */}
      <div className="overflow-x-auto p-4">
        <pre className="text-sm leading-relaxed">
          <code
            className="text-gray-300 font-mono"
            dangerouslySetInnerHTML={{ __html: highlightCode(code) }}
          />
        </pre>
      </div>
    </div>
  );
}
