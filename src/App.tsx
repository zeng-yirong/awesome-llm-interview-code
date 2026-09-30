import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Code2, Zap, BookOpen, ExternalLink, Sparkles, Github, Star } from 'lucide-react';
import ProblemCard from './components/ProblemCard';
import SearchFilter from './components/SearchFilter';
import { problems, categories, sources, Category, Difficulty } from './data/problems';

export default function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
  const [selectedDifficulty, setSelectedDifficulty] = useState<Difficulty | null>(null);

  const filteredProblems = useMemo(() => {
    return problems.filter((problem) => {
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesSearch =
          problem.title.toLowerCase().includes(query) ||
          problem.titleCn.includes(query) ||
          problem.description.includes(query) ||
          problem.tags.some((tag) => tag.toLowerCase().includes(query)) ||
          problem.keyPoints.some((point) => point.includes(query));
        if (!matchesSearch) return false;
      }
      if (selectedCategory && problem.category !== selectedCategory) return false;
      if (selectedDifficulty && problem.difficulty !== selectedDifficulty) return false;
      return true;
    });
  }, [searchQuery, selectedCategory, selectedDifficulty]);

  const stats = useMemo(() => ({
    total: problems.length,
    easy: problems.filter((p) => p.difficulty === 'Easy').length,
    medium: problems.filter((p) => p.difficulty === 'Medium').length,
    hard: problems.filter((p) => p.difficulty === 'Hard').length,
  }), []);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/3 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-gray-800/50">
          <div className="max-w-6xl mx-auto px-4 py-8 sm:py-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center"
            >
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                    <Brain size={24} className="text-white" />
                  </div>
                  <Sparkles size={14} className="absolute -top-1 -right-1 text-yellow-400" />
                </div>
              </div>

              <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent mb-3">
                Hand-torn Code for LLM Interviews
              </h1>
              <p className="text-gray-400 text-lg max-w-2xl mx-auto mb-6">
                大模型面试手撕代码总结 — 整合多个优质开源仓库，涵盖注意力机制、归一化、位置编码、FFN、损失函数、参数高效微调等核心主题
              </p>

              {/* Stats */}
              <div className="flex items-center justify-center gap-6 flex-wrap">
                <div className="flex items-center gap-2 text-sm">
                  <Code2 size={16} className="text-blue-400" />
                  <span className="text-gray-400">共</span>
                  <span className="text-white font-bold">{stats.total}</span>
                  <span className="text-gray-400">题</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <span className="w-2 h-2 rounded-full bg-green-400" />
                  <span className="text-gray-400">Easy</span>
                  <span className="text-white font-bold">{stats.easy}</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <span className="w-2 h-2 rounded-full bg-yellow-400" />
                  <span className="text-gray-400">Medium</span>
                  <span className="text-white font-bold">{stats.medium}</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <span className="w-2 h-2 rounded-full bg-red-400" />
                  <span className="text-gray-400">Hard</span>
                  <span className="text-white font-bold">{stats.hard}</span>
                </div>
              </div>
            </motion.div>
          </div>
        </header>

        {/* Source Repos */}
        <div className="max-w-6xl mx-auto px-4 pt-6">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-8"
          >
            {Object.entries(sources).map(([key, source]) => (
              <a
                key={key}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 p-3 rounded-xl bg-gray-900/50 border border-gray-800 hover:border-gray-600 transition-all group"
              >
                <div className="w-8 h-8 rounded-lg bg-gray-800 flex items-center justify-center flex-shrink-0">
                  <Github size={16} className="text-gray-400 group-hover:text-white transition-colors" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">{source.name}</p>
                  <p className="text-xs text-gray-500 truncate">{source.description}</p>
                </div>
                <div className="flex items-center gap-1 text-xs text-yellow-500 flex-shrink-0">
                  <Star size={12} fill="currentColor" />
                  <span>{source.stars}</span>
                </div>
                <ExternalLink size={14} className="text-gray-600 group-hover:text-gray-400 transition-colors flex-shrink-0" />
              </a>
            ))}
          </motion.div>
        </div>

        {/* Main Content */}
        <main className="max-w-6xl mx-auto px-4 pb-8">
          {/* Search & Filter */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 }}
            className="mb-8"
          >
            <SearchFilter
              searchQuery={searchQuery}
              onSearchChange={setSearchQuery}
              selectedCategory={selectedCategory}
              onCategoryChange={setSelectedCategory}
              selectedDifficulty={selectedDifficulty}
              onDifficultyChange={setSelectedDifficulty}
              resultCount={filteredProblems.length}
            />
          </motion.div>

          {/* Category Quick Nav */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mb-8 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3"
          >
            {categories.slice(0, 6).map((cat) => {
              const count = problems.filter((p) => p.category === cat.name).length;
              if (count === 0) return null;
              return (
                <button
                  key={cat.name}
                  onClick={() => setSelectedCategory(selectedCategory === cat.name ? null : cat.name)}
                  className={`p-3 rounded-xl border text-left transition-all ${
                    selectedCategory === cat.name
                      ? 'bg-blue-500/10 border-blue-500/30'
                      : 'bg-gray-900/50 border-gray-800 hover:border-gray-700'
                  }`}
                >
                  <span className="text-xl">{cat.icon}</span>
                  <p className="text-sm font-medium text-white mt-1">{cat.name}</p>
                  <p className="text-xs text-gray-500">{count} 题</p>
                </button>
              );
            })}
          </motion.div>

          {/* Problem List */}
          <div className="space-y-4">
            <AnimatePresence mode="popLayout">
              {filteredProblems.length > 0 ? (
                filteredProblems.map((problem) => (
                  <ProblemCard key={problem.id} problem={problem} />
                ))
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-16"
                >
                  <BookOpen size={48} className="mx-auto text-gray-600 mb-4" />
                  <p className="text-gray-400 text-lg">没有找到匹配的题目</p>
                  <p className="text-gray-500 text-sm mt-1">尝试调整搜索条件或筛选器</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-800/50 mt-16">
          <div className="max-w-6xl mx-auto px-4 py-8">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-gray-500 text-sm">
                <Zap size={14} className="text-yellow-400" />
                <span>整合自多个优质开源仓库 · 持续更新中 · 面试加油 💪</span>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-500">
                <a
                  href={sources.ckd0817.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-gray-300 transition-colors"
                >
                  ckd0817/LLM-Interview-Code
                </a>
                <span>·</span>
                <a
                  href={sources.cdhx.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-gray-300 transition-colors"
                >
                  cdhx/LLM-Code-Hot-100
                </a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
