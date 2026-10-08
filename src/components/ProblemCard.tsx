import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, BookOpen, Lightbulb, Flame, Star, Code2, GitBranch } from 'lucide-react';
import CodeBlock from './CodeBlock';
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
      className="bg-gray-900/60 backdrop-blur-sm border border-gray-800 rounded-xl overflow-hidden hover:border-gray-700 transition-colors"
    >
      {/* Header */}
      <div className="p-5 cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            {/* Badges row */}
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              {/* Hot */}
              <span className="inline-flex items-center gap-0.5 text-xs px-2 py-0.5 rounded-full bg-orange-500/15 text-orange-400 border border-orange-500/25">
                {Array.from({ length: problem.hot }).map((_, i) => (
                  <Flame key={i} size={10} fill="currentColor" />
                ))}
              </span>
              {/* Difficulty */}
              <span className="inline-flex items-center gap-0.5 text-xs px-2 py-0.5 rounded-full bg-yellow-500/15 text-yellow-400 border border-yellow-500/25">
                {Array.from({ length: problem.difficulty }).map((_, i) => (
                  <Star key={i} size={10} fill="currentColor" />
                ))}
                {Array.from({ length: 5 - problem.difficulty }).map((_, i) => (
                  <Star key={`e${i}`} size={10} />
                ))}
              </span>
              {/* Category */}
              <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 border border-gray-700">
                {problem.category}
              </span>
            </div>
            {/* Title */}
            <h3 className="text-lg font-semibold text-white mb-0.5">{problem.titleCn}</h3>
            <p className="text-sm text-gray-500">{problem.title}</p>
          </div>
          <motion.div
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-gray-500 mt-1 flex-shrink-0"
          >
            <ChevronDown size={20} />
          </motion.div>
        </div>
        {/* One-liner */}
        <p className="text-sm text-gray-300 mt-3 font-medium">{problem.oneLiner}</p>
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
            <div className="px-5 pb-5 space-y-4 border-t border-gray-800 pt-4">
              {/* Principle */}
              <div className="bg-gray-800/40 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Lightbulb size={15} className="text-amber-400" />
                  <h4 className="text-sm font-semibold text-amber-400">原理 & 思想</h4>
                </div>
                <p className="text-sm text-gray-300 leading-relaxed">{problem.principle}</p>
              </div>

              {/* Formula */}
              <div className="bg-gray-800/40 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm">📐</span>
                  <h4 className="text-sm font-semibold text-cyan-400">核心公式</h4>
                </div>
                <pre className="text-sm text-cyan-300 font-mono whitespace-pre-wrap leading-relaxed overflow-x-auto">
                  {problem.formula}
                </pre>
              </div>

              {/* Flow Diagram */}
              <div className="bg-gray-800/40 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <GitBranch size={15} className="text-green-400" />
                  <h4 className="text-sm font-semibold text-green-400">张量流程图</h4>
                </div>
                <pre className="text-xs text-green-300/90 font-mono whitespace-pre overflow-x-auto leading-relaxed">
                  {problem.flowDiagram}
                </pre>
              </div>

              {/* Code */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Code2 size={15} className="text-blue-400" />
                  <h4 className="text-sm font-semibold text-blue-400">代码实现</h4>
                </div>
                <CodeBlock code={problem.code} language="python" />
              </div>

              {/* Key Points */}
              <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <BookOpen size={15} className="text-amber-400" />
                  <h4 className="text-sm font-semibold text-amber-400">面试要点</h4>
                </div>
                <ul className="space-y-1.5">
                  {problem.keyPoints.map((point, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-300">
                      <span className="text-amber-500 mt-0.5 flex-shrink-0">▸</span>
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
