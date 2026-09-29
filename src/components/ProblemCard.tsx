import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, BookOpen, Lightbulb } from 'lucide-react';
import CodeBlock from './CodeBlock';
import { Problem } from '../data/problems';

interface ProblemCardProps {
  problem: Problem;
}

const difficultyColors = {
  Easy: 'bg-green-500/20 text-green-400 border-green-500/30',
  Medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  Hard: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const categoryColors: Record<string, string> = {
  Transformer: 'bg-blue-500/20 text-blue-400',
  Attention: 'bg-purple-500/20 text-purple-400',
  Tokenizer: 'bg-emerald-500/20 text-emerald-400',
  Training: 'bg-orange-500/20 text-orange-400',
  Inference: 'bg-cyan-500/20 text-cyan-400',
  'Data Structures': 'bg-pink-500/20 text-pink-400',
  Algorithms: 'bg-indigo-500/20 text-indigo-400',
  'System Design': 'bg-rose-500/20 text-rose-400',
};

export default function ProblemCard({ problem }: ProblemCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-gray-900/50 backdrop-blur-sm border border-gray-800 rounded-xl overflow-hidden hover:border-gray-700 transition-colors"
    >
      {/* Header */}
      <div
        className="p-5 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <span className={`text-xs px-2 py-0.5 rounded-full border ${difficultyColors[problem.difficulty]}`}>
                {problem.difficulty}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${categoryColors[problem.category] || 'bg-gray-500/20 text-gray-400'}`}>
                {problem.category}
              </span>
            </div>
            <h3 className="text-lg font-semibold text-white mb-1">
              {problem.titleCn}
            </h3>
            <p className="text-sm text-gray-400">
              {problem.title}
            </p>
          </div>
          <motion.div
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-gray-400 mt-1 flex-shrink-0"
          >
            <ChevronDown size={20} />
          </motion.div>
        </div>

        <p className="text-sm text-gray-400 mt-3 line-clamp-2">
          {problem.description}
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mt-3">
          {problem.tags.map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400 border border-gray-700"
            >
              #{tag}
            </span>
          ))}
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
            <div className="px-5 pb-5 space-y-4 border-t border-gray-800 pt-4">
              {/* Key Points */}
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Lightbulb size={16} className="text-yellow-400" />
                  <h4 className="text-sm font-semibold text-yellow-400">面试要点</h4>
                </div>
                <ul className="space-y-2">
                  {problem.keyPoints.map((point, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-gray-300">
                      <span className="text-yellow-400 mt-0.5 flex-shrink-0">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Code */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <BookOpen size={16} className="text-blue-400" />
                  <h4 className="text-sm font-semibold text-blue-400">参考实现</h4>
                </div>
                <CodeBlock code={problem.code} language={problem.language} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
