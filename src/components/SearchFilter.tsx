import { Search, Filter, X } from 'lucide-react';
import { Category, Difficulty, categories, difficulties } from '../data/problems';

interface SearchFilterProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  selectedCategory: Category | null;
  onCategoryChange: (category: Category | null) => void;
  selectedDifficulty: Difficulty | null;
  onDifficultyChange: (difficulty: Difficulty | null) => void;
  resultCount: number;
}

const difficultyColors = {
  Easy: 'bg-green-500/20 text-green-400 border-green-500/30 hover:bg-green-500/30',
  Medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30 hover:bg-yellow-500/30',
  Hard: 'bg-red-500/20 text-red-400 border-red-500/30 hover:bg-red-500/30',
};

const difficultyActiveColors = {
  Easy: 'bg-green-500/40 text-green-300 border-green-400',
  Medium: 'bg-yellow-500/40 text-yellow-300 border-yellow-400',
  Hard: 'bg-red-500/40 text-red-300 border-red-400',
};

export default function SearchFilter({
  searchQuery,
  onSearchChange,
  selectedCategory,
  onCategoryChange,
  selectedDifficulty,
  onDifficultyChange,
  resultCount,
}: SearchFilterProps) {
  const hasFilters = searchQuery || selectedCategory || selectedDifficulty;

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="relative">
        <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" />
        <input
          type="text"
          placeholder="搜索题目、标签、描述..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full pl-11 pr-10 py-3 bg-gray-900/80 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="space-y-3">
        {/* Category Filter */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Filter size={14} className="text-gray-500" />
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">分类</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {categories.map((cat) => (
              <button
                key={cat.name}
                onClick={() => onCategoryChange(selectedCategory === cat.name ? null : cat.name)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                  selectedCategory === cat.name
                    ? 'bg-blue-500/30 text-blue-300 border-blue-500/50'
                    : 'bg-gray-800/50 text-gray-400 border-gray-700 hover:bg-gray-800 hover:text-gray-300'
                }`}
              >
                <span className="mr-1">{cat.icon}</span>
                {cat.name}
              </button>
            ))}
          </div>
        </div>

        {/* Difficulty Filter */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Filter size={14} className="text-gray-500" />
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">难度</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {difficulties.map((diff) => (
              <button
                key={diff}
                onClick={() => onDifficultyChange(selectedDifficulty === diff ? null : diff)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                  selectedDifficulty === diff
                    ? difficultyActiveColors[diff]
                    : difficultyColors[diff]
                }`}
              >
                {diff}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results count and clear */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-800">
        <span className="text-sm text-gray-400">
          共 <span className="text-white font-medium">{resultCount}</span> 道题目
        </span>
        {hasFilters && (
          <button
            onClick={() => {
              onSearchChange('');
              onCategoryChange(null);
              onDifficultyChange(null);
            }}
            className="text-xs text-gray-400 hover:text-white transition-colors flex items-center gap-1"
          >
            <X size={12} />
            清除筛选
          </button>
        )}
      </div>
    </div>
  );
}
