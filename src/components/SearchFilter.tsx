import { Search, X, Flame, Star } from 'lucide-react';
import { Category, Difficulty, categories } from '../data/problems';

interface SearchFilterProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  selectedCategory: Category | null;
  onCategoryChange: (category: Category | null) => void;
  selectedHot: number | null;
  onHotChange: (hot: number | null) => void;
  selectedDifficulty: number | null;
  onDifficultyChange: (difficulty: number | null) => void;
  resultCount: number;
}

const hotLevels = [3, 2, 1] as const;
const diffLevels = [5, 4, 3, 2, 1] as const;

export default function SearchFilter({
  searchQuery, onSearchChange,
  selectedCategory, onCategoryChange,
  selectedHot, onHotChange,
  selectedDifficulty, onDifficultyChange,
  resultCount,
}: SearchFilterProps) {
  const hasFilters = searchQuery || selectedCategory || selectedHot !== null || selectedDifficulty !== null;

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="relative">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" />
        <input
          type="text"
          placeholder="搜索题目、原理、公式、要点..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full pl-11 pr-10 py-3 bg-gray-900/80 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
        />
        {searchQuery && (
          <button onClick={() => onSearchChange('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Category */}
        <div className="md:col-span-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">分类</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {categories.map((cat) => (
              <button key={cat.name}
                onClick={() => onCategoryChange(selectedCategory === cat.name ? null : cat.name)}
                className={`text-xs px-2.5 py-1.5 rounded-lg border transition-all ${
                  selectedCategory === cat.name
                    ? 'bg-blue-500/30 text-blue-300 border-blue-500/50'
                    : 'bg-gray-800/50 text-gray-400 border-gray-700 hover:bg-gray-800 hover:text-gray-300'
                }`}>
                <span className="mr-1">{cat.icon}</span>{cat.name}
              </button>
            ))}
          </div>
        </div>

        {/* Hot */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Flame size={12} className="text-orange-400" />
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">热度</span>
          </div>
          <div className="flex gap-2">
            {hotLevels.map((h) => (
              <button key={h}
                onClick={() => onHotChange(selectedHot === h ? null : h)}
                className={`flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg border transition-all ${
                  selectedHot === h
                    ? 'bg-orange-500/30 text-orange-300 border-orange-500/50'
                    : 'bg-gray-800/50 text-gray-400 border-gray-700 hover:bg-gray-800'
                }`}>
                {Array.from({ length: h }).map((_, i) => <Flame key={i} size={10} fill="currentColor" />)}
              </button>
            ))}
          </div>
        </div>

        {/* Difficulty */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Star size={12} className="text-yellow-400" />
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">难度</span>
          </div>
          <div className="flex gap-2">
            {diffLevels.map((d) => (
              <button key={d}
                onClick={() => onDifficultyChange(selectedDifficulty === d ? null : d)}
                className={`flex items-center gap-0.5 text-xs px-2 py-1.5 rounded-lg border transition-all ${
                  selectedDifficulty === d
                    ? 'bg-yellow-500/30 text-yellow-300 border-yellow-500/50'
                    : 'bg-gray-800/50 text-gray-400 border-gray-700 hover:bg-gray-800'
                }`}>
                {Array.from({ length: d }).map((_, i) => <Star key={i} size={9} fill="currentColor" />)}
              </button>
            ))}
          </div>
        </div>

        {/* Results */}
        <div className="flex items-end justify-end">
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400">
              <span className="text-white font-bold">{resultCount}</span> 题
            </span>
            {hasFilters && (
              <button onClick={() => { onSearchChange(''); onCategoryChange(null); onHotChange(null); onDifficultyChange(null); }}
                className="text-xs text-gray-500 hover:text-white transition-colors flex items-center gap-1">
                <X size={12} />清除
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
