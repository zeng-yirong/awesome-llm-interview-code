import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Lightbulb, Code2, GitBranch, BookOpen } from 'lucide-react';
import CodeBlock from './CodeBlock';
import FlowDiagram from './FlowDiagram';
import { Problem } from '../data/problems';

interface ProblemCardProps {
  problem: Problem;
}

export default function ProblemCard({ problem }: ProblemCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="rounded-xl border overflow-hidden transition-all hover:shadow-lg"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border-color)'
      }}
    >
      {/* Header */}
      <div 
        className="p-6 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            {/* Badges */}
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              {/* Hot */}
              <span 
                className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full font-medium"
                style={{
                  backgroundColor: 'rgba(239, 68, 68, 0.1)',
                  color: '#ef4444'
                }}
              >
                {'🔥'.repeat(problem.hot)}
              </span>
              
              {/* Difficulty */}
              <span 
                className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full font-medium"
                style={{
                  backgroundColor: 'rgba(234, 179, 8, 0.1)',
                  color: '#eab308'
                }}
              >
                {'⭐'.repeat(problem.difficulty)}
              </span>
              
              {/* Category */}
              <span 
                className="text-xs px-2 py-1 rounded-full font-medium"
                style={{
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-secondary)'
                }}
              >
                {problem.category}
              </span>
            </div>

            {/* Title */}
            <h3 
              className="text-xl font-bold mb-1"
              style={{ color: 'var(--text-primary)' }}
            >
              {problem.titleCn}
            </h3>
            <p 
              className="text-sm mb-3"
              style={{ color: 'var(--text-tertiary)' }}
            >
              {problem.title}
            </p>

            {/* One-liner */}
            <p 
              className="text-sm font-medium"
              style={{ color: 'var(--text-secondary)' }}
            >
              {problem.oneLiner}
            </p>
          </div>

          <motion.div
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            style={{ color: 'var(--text-tertiary)' }}
          >
            <ChevronDown size={24} />
          </motion.div>
        </div>
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div 
              className="px-6 pb-6 space-y-6 border-t"
              style={{ borderColor: 'var(--border-color)' }}
            >
              {/* Principle */}
              <div className="pt-6">
                <div className="flex items-center gap-2 mb-3">
                  <Lightbulb size={18} style={{ color: '#f59e0b' }} />
                  <h4 
                    className="text-sm font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    原理 & 思想
                  </h4>
                </div>
                <p
                  className="text-sm leading-relaxed"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {problem.principle}
                </p>

                {/* 原理分节展开：动机 / 直觉 / 步骤 / 代价 */}
                {problem.principleSections && problem.principleSections.length > 0 && (
                  <div className="mt-4 space-y-4">
                    {problem.principleSections.map((section) => (
                      <div key={section.title}>
                        <h5
                          className="text-xs font-semibold mb-2"
                          style={{ color: 'var(--text-primary)' }}
                        >
                          {section.title}
                        </h5>
                        <ul className="space-y-2">
                          {section.items.map((item, idx) => (
                            <li
                              key={idx}
                              className="flex items-start gap-2 text-sm"
                              style={{ color: 'var(--text-secondary)' }}
                            >
                              <span style={{ color: '#f59e0b' }}>▸</span>
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Formula */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-lg">📐</span>
                  <h4 
                    className="text-sm font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    核心公式
                  </h4>
                </div>
                <pre 
                  className="text-sm font-mono whitespace-pre-wrap p-4 rounded-lg overflow-x-auto"
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)'
                  }}
                >
                  {problem.formula}
                </pre>
              </div>

              {/* Flow Diagram */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <GitBranch size={18} style={{ color: '#10b981' }} />
                  <h4 
                    className="text-sm font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    张量流程图
                  </h4>
                </div>
                <FlowDiagram source={problem.flowDiagram} />
              </div>

              {/* Code */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Code2 size={18} style={{ color: 'var(--accent-color)' }} />
                  <h4 
                    className="text-sm font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    代码实现
                  </h4>
                </div>
                <CodeBlock code={problem.code} language="python" />
              </div>

              {/* Key Points */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <BookOpen size={18} style={{ color: '#f59e0b' }} />
                  <h4 
                    className="text-sm font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    面试要点
                  </h4>
                </div>
                <ul className="space-y-2">
                  {problem.keyPoints.map((point, idx) => (
                    <li 
                      key={idx}
                      className="flex items-start gap-2 text-sm"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      <span style={{ color: '#f59e0b' }}>▸</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
