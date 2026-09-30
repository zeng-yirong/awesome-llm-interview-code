import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Code2, Zap, BookOpen, ExternalLink, Sparkles, Github, Star, Flame } from 'lucide-react';
import ProblemCard from './components/ProblemCard';
import SearchFilter from './components/SearchFilter';
import { problems, categories, sources, Category } from './data/problems';

export default function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
  const [selectedHot, setSelectedHot] = useState<number | null>(null);
  const [selectedDifficulty, setSelectedDifficulty] = useState<number | null>(null);
  const [sortBy, setSortBy] = useState<'hot' | 'difficulty' | 'category'>('hot');

  const filteredProblems = useMemo(() => {
    let result = problems.filter((p) => {
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const match =
          p.title.toLowerCase().includes(q) ||
          p.titleCn.includes(q) ||
          p.oneLiner.includes(q) ||
          p.principle.includes(q) ||
          p.formula.toLowerCase().includes(q) ||
          p.keyPoints.some(kp => kp.includes(q)) ||
          p.category.toLowerCase().includes(q);
        if (!match) return false;
      }
      if (selectedCategory && p.category !== selectedCategory) return false;
      if (selectedHot !== null && p.hot !== selectedHot) return false;
      if (selectedDifficulty !== null && p.difficulty !== selectedDifficulty) return false;
      return true;
    });

    // Sort
    if (sortBy === 'hot') result.sort((a, b) => b.hot - a.hot || b.difficulty - a.difficulty);
    else if (sortBy === 'difficulty') result.sort((a, b) => b.difficulty - a.difficulty);
    else result.sort((a, b) => a.category.localeCompare(b.category));

    return result;
  }, [searchQuery, selectedCategory, selectedHot, selectedDifficulty, sortBy]);

  const stats = useMemo(() => ({
    total: problems.length,
    hot3: problems.filter(p => p.hot === 3).length,
    hot2: problems.filter(p => p.hot === 2).length,
    hot1: problems.filter(p => p.hot === 1).length,
  }), []);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* BG */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-gray-800/50">
          <div className="max-w-6xl mx-auto px-4 py-8 sm:py-10">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center">
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className="relative">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                    <Brain size={24} className="text-white" />
                  </div>
                  <Sparkles size={14} className="absolute -top-1 -right-1 text-yellow-400" />
                </div>
              </div>
              <h1 className="text-2xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent mb-2">
                Hand-torn Code for LLM Interviews
              </h1>
              <p className="text-gray-400 text-base max-w-2xl mx-auto mb-5">
                整合两大优质开源仓库，去重统一风格 — 每题含原理、公式、流程图、代码、面试要点
              </p>

              {/* Stats bar */}
              <div className="flex items-center justify-center gap-4 flex-wrap text-sm">
                <div className="flex items-center gap-1.5">
                  <Code2 size={14} className="text-blue-400" />
                  <span className="text-gray-400">共</span>
                  <span className="text-white font-bold">{stats.total}</span>
                  <span className="text-gray-400">题</span>
                </div>
                <div className="h-4 w-px bg-gray-700" />
                <div className="flex items-center gap-1">
                  <Flame size={12} className="text-orange-400" fill="currentColor" />
                  <Flame size={12} className="text-orange-400" fill="currentColor" />
                  <Flame size={12} className="text-orange-400" fill="currentColor" />
                  <span className="text-white font-bold ml-0.5">{stats.hot3}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Flame size={12} className="text-orange-400" fill="currentColor" />
                  <Flame size={12} className="text-orange-400" fill="currentColor" />
                  <span className="text-white font-bold ml-0.5">{stats.hot2}</span>
                </div>
                <div className="h-4 w-px bg-gray-700" />
                <div className="flex items-center gap-1">
                  <Star size={12} className="text-yellow-400" fill="currentColor" />
                  <span className="text-gray-400">最高难度</span>
                  <span className="text-white font-bold">5</span>
                </div>
              </div>
            </motion.div>
          </div>
        </header>

        {/* Source Repos */}
        <div className="max-w-6xl mx-auto px-4 pt-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-6">
            {Object.entries(sources).map(([key, source]) => (
              <a key={key} href={source.url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-3 p-2.5 rounded-lg bg-gray-900/50 border border-gray-800 hover:border-gray-600 transition-all group">
                <Github size={16} className="text-gray-400 group-hover:text-white flex-shrink-0" />
                <span className="text-sm text-gray-300 truncate flex-1">{source.name}</span>
                <div className="flex items-center gap-1 text-xs text-yellow-500 flex-shrink-0">
                  <Star size={11} fill="currentColor" />{source.stars}
                </div>
                <ExternalLink size={12} className="text-gray-600 group-hover:text-gray-400 flex-shrink-0" />
              </a>
            ))}
          </div>
        </div>

        {/* Main */}
        <main className="max-w-6xl mx-auto px-4 pb-8">
          {/* Search & Filter */}
          <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mb-6">
            <SearchFilter
              searchQuery={searchQuery} onSearchChange={setSearchQuery}
              selectedCategory={selectedCategory} onCategoryChange={setSelectedCategory}
              selectedHot={selectedHot} onHotChange={setSelectedHot}
              selectedDifficulty={selectedDifficulty} onDifficultyChange={setSelectedDifficulty}
              resultCount={filteredProblems.length}
            />
          </motion.div>

          {/* Sort & Category Quick Nav */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 overflow-x-auto pb-1">
              {categories.slice(0, 7).map((cat) => {
                const count = problems.filter(p => p.category === cat.name).length;
                if (count === 0) return null;
                return (
                  <button key={cat.name}
                    onClick={() => setSelectedCategory(selectedCategory === cat.name ? null : cat.name)}
                    className={`flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg border whitespace-nowrap transition-all ${
                      selectedCategory === cat.name
                        ? 'bg-blue-500/15 border-blue-500/30 text-blue-300'
                        : 'bg-gray-900/50 border-gray-800 text-gray-400 hover:border-gray-700'
                    }`}>
                    <span>{cat.icon}</span>
                    <span>{cat.name}</span>
                    <span className="text-gray-600">{count}</span>
                  </button>
                );
              })}
            </div>
            <div className="flex items-center gap-1 flex-shrink-0 ml-2">
              <span className="text-xs text-gray-500 mr-1">排序:</span>
              {(['hot', 'difficulty', 'category'] as const).map(s => (
                <button key={s} onClick={() => setSortBy(s)}
                  className={`text-xs px-2 py-1 rounded transition-all ${
                    sortBy === s ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'
                  }`}>
                  {s === 'hot' ? '🔥热度' : s === 'difficulty' ? '⭐难度' : '📂分类'}
                </button>
              ))}
            </div>
          </div>

          {/* Problem List */}
          <div className="space-y-3">
            <AnimatePresence mode="popLayout">
              {filteredProblems.length > 0 ? (
                filteredProblems.map((problem) => (
                  <ProblemCard key={problem.id} problem={problem} />
                ))
              ) : (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-16">
                  <BookOpen size={48} className="mx-auto text-gray-700 mb-4" />
                  <p className="text-gray-400">没有找到匹配的题目</p>
                  <p className="text-gray-600 text-sm mt-1">尝试调整搜索条件</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-800/50 mt-12">
          <div className="max-w-6xl mx-auto px-4 py-6 flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-gray-500 text-xs">
              <Zap size={12} className="text-yellow-400" />
              <span>整合自两个优质开源仓库 · 去重统一风格 · 持续更新</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-gray-600">
              <a href={sources.ckd0817.url} target="_blank" rel="noopener noreferrer" className="hover:text-gray-400">ckd0817</a>
              <span>·</span>
              <a href={sources.cdhx.url} target="_blank" rel="noopener noreferrer" className="hover:text-gray-400">cdhx</a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
