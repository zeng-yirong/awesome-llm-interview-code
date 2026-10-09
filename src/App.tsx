import { useState, useMemo } from 'react';
import { ThemeProvider, useTheme } from './contexts/ThemeContext';
import { problems, categories, Category } from './data/problems';
import ProblemCard from './components/ProblemCard';
import SearchFilter from './components/SearchFilter';
import { plainMath } from './lib/mathBlock';
import { Sun, Moon, BookOpen } from 'lucide-react';

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <button
      onClick={toggleTheme}
      className="fixed top-6 right-6 p-2 rounded-lg border transition-all hover:scale-110 z-50"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border-color)',
        color: 'var(--text-primary)'
      }}
      aria-label="Toggle theme"
    >
      {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
    </button>
  );
}

function AppContent() {
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
          // formula 现在是 LaTeX，先投影成纯文本，否则 \text{softmax} 这类搜不到
          plainMath(p.formula).toLowerCase().includes(q) ||
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

  return (
    <div className="min-h-screen">
      <ThemeToggle />
      
      {/* Header */}
      <header 
        className="border-b sticky top-0 z-10 backdrop-blur-sm"
        style={{
          backgroundColor: 'var(--bg-primary)',
          borderColor: 'var(--border-color)'
        }}
      >
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-center gap-3 mb-2">
            <BookOpen size={32} style={{ color: 'var(--accent-color)' }} />
            <h1 
              className="text-3xl font-bold"
              style={{ color: 'var(--text-primary)' }}
            >
              LLM 面试手撕代码
            </h1>
          </div>
          <p 
            className="text-base"
            style={{ color: 'var(--text-secondary)' }}
          >
            大模型面试高频代码题汇总 · 原理 · 公式 · 代码 · 要点
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Search and Filter */}
        <div className="mb-8">
          <SearchFilter
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            selectedCategory={selectedCategory}
            onCategoryChange={setSelectedCategory}
            selectedHot={selectedHot}
            onHotChange={setSelectedHot}
            selectedDifficulty={selectedDifficulty}
            onDifficultyChange={setSelectedDifficulty}
            resultCount={filteredProblems.length}
          />
        </div>

        {/* Category Quick Nav */}
        <div className="mb-6 flex flex-wrap gap-2">
          {categories.map(category => (
            <button
              key={category.name}
              onClick={() => setSelectedCategory(
                selectedCategory === category.name ? null : category.name
              )}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all hover:scale-105"
              style={{
                backgroundColor: selectedCategory === category.name 
                  ? 'var(--accent-color)' 
                  : 'var(--bg-secondary)',
                color: selectedCategory === category.name 
                  ? '#ffffff' 
                  : 'var(--text-primary)',
                border: `1px solid ${
                  selectedCategory === category.name 
                    ? 'var(--accent-color)' 
                    : 'var(--border-color)'
                }`
              }}
            >
              {category.icon} {category.name}
            </button>
          ))}
        </div>

        {/* Sort Options */}
        <div className="mb-6 flex items-center gap-4">
          <span style={{ color: 'var(--text-secondary)' }}>排序：</span>
          {(['hot', 'difficulty', 'category'] as const).map(option => (
            <button
              key={option}
              onClick={() => setSortBy(option)}
              className="px-3 py-1 rounded text-sm transition-all"
              style={{
                backgroundColor: sortBy === option ? 'var(--accent-color)' : 'transparent',
                color: sortBy === option ? '#ffffff' : 'var(--text-secondary)',
              }}
            >
              {option === 'hot' ? '🔥 热度' : option === 'difficulty' ? '⭐ 难度' : '📂 分类'}
            </button>
          ))}
        </div>

        {/* Problem List */}
        <div className="space-y-4">
          {filteredProblems.length === 0 ? (
            <div 
              className="text-center py-16"
              style={{ color: 'var(--text-tertiary)' }}
            >
              没有找到匹配的题目
            </div>
          ) : (
            filteredProblems.map(problem => (
              <ProblemCard key={problem.id} problem={problem} />
            ))
          )}
        </div>
      </main>

      {/* Footer */}
      <footer 
        className="border-t mt-16"
        style={{
          borderColor: 'var(--border-color)',
          backgroundColor: 'var(--bg-secondary)'
        }}
      >
        <div className="max-w-6xl mx-auto px-6 py-8 text-center">
          <p style={{ color: 'var(--text-tertiary)' }}>
            © 2024 LLM Interview Code · 持续更新中
          </p>
        </div>
      </footer>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
