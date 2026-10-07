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
    
    // 第二步：构建综合正则表达式，一次性匹配所有模式
    // 使用捕获组来识别匹配的类型
    const combinedPattern = new RegExp(
      '(#[^\\n]*)' +                    // 组1: 注释
      '|("""[\\s\\S]*?""")' +           // 组2: 三引号字符串
      "|('[^']*')" +                    // 组3: 单引号字符串
      '|("[^"]*")' +                    // 组4: 双引号字符串
      '|(@\\w+)' +                      // 组5: 装饰器
      '|\\b(import|from|class|def|return|if|else|elif|for|while|in|not|and|or|is|None|True|False|self|super|with|as|try|except|raise|pass|break|continue|lambda|yield|assert|global|nonlocal)\\b' + // 组6: 关键字
      '|\\b(int|float|str|bool|list|dict|tuple|Optional|List|Dict|Tuple|Callable|Any|Union)\\b' + // 组7: 类型
      '|\\b(\\d+\\.?\\d*)\\b' +         // 组8: 数字
      '|\\b(torch|nn|F|math)\\b',       // 组9: torch相关
      'gm'
    );
    
    // 第三步：一次性替换所有匹配
    escaped = escaped.replace(combinedPattern, (match, p1, p2, p3, p4, p5, p6, p7, p8, p9) => {
      if (p1 !== undefined) return `<span class="text-gray-500 italic">${p1}</span>`;
      if (p2 !== undefined) return `<span class="text-gray-500">${p2}</span>`;
      if (p3 !== undefined) return `<span class="text-emerald-400">${p3}</span>`;
      if (p4 !== undefined) return `<span class="text-emerald-400">${p4}</span>`;
      if (p5 !== undefined) return `<span class="text-yellow-400">${p5}</span>`;
      if (p6 !== undefined) return `<span class="text-purple-400 font-medium">${p6}</span>`;
      if (p7 !== undefined) return `<span class="text-cyan-400">${p7}</span>`;
      if (p8 !== undefined) return `<span class="text-orange-400">${p8}</span>`;
      if (p9 !== undefined) return `<span class="text-rose-400">${p9}</span>`;
      return match;
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
