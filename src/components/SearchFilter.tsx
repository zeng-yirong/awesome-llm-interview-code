import { Search, X } from 'lucide-react';
import { Category, categories } from '../data/problems';

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

export default function SearchFilter({
  searchQuery,
  onSearchChange,
  selectedHot,
  onHotChange,
  selectedDifficulty,
  onDifficultyChange,
  resultCount,
}: SearchFilterProps) {
  const hasFilters = searchQuery || selectedHot !== null || selectedDifficulty !== null;

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="relative">
        <Search 
          size={20} 
          className="absolute left-4 top-1/2 -translate-y-1/2"
          style={{ color: 'var(--text-tertiary)' }}
        />
        <input
          type="text"
          placeholder="搜索题目、原理、公式..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full pl-12 pr-12 py-3 rounded-lg border text-base transition-all focus:outline-none focus:ring-2 focus:ring-blue-500"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderColor: 'var(--border-color)',
            color: 'var(--text-primary)'
          }}
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange('')}
            className="absolute right-4 top-1/2 -translate-y-1/2 p-1 rounded hover:opacity-70"
            style={{ color: 'var(--text-tertiary)' }}
          >
            <X size={20} />
          </button>
        )}
      </div>

      {/* Filters Row */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Hot Filter */}
        <div className="flex items-center gap-2">
          <span 
            className="text-sm font-medium"
            style={{ color: 'var(--text-secondary)' }}
          >
            热度：
          </span>
          <div className="flex gap-2">
            {[3, 2, 1].map(hot => (
              <button
                key={hot}
                onClick={() => onHotChange(selectedHot === hot ? null : hot)}
                className="px-3 py-1 rounded text-sm transition-all hover:scale-105"
                style={{
                  backgroundColor: selectedHot === hot 
                    ? 'rgba(239, 68, 68, 0.2)' 
                    : 'var(--bg-secondary)',
                  color: selectedHot === hot 
                    ? '#ef4444' 
                    : 'var(--text-secondary)',
                  border: `1px solid ${
                    selectedHot === hot 
                      ? 'rgba(239, 68, 68, 0.3)' 
                      : 'var(--border-color)'
                  }`
                }}
              >
                {'🔥'.repeat(hot)}
              </button>
            ))}
          </div>
        </div>

        {/* Difficulty Filter */}
        <div className="flex items-center gap-2">
          <span 
            className="text-sm font-medium"
            style={{ color: 'var(--text-secondary)' }}
          >
            难度：
          </span>
          <div className="flex gap-2">
            {[5, 4, 3, 2, 1].map(diff => (
              <button
                key={diff}
                onClick={() => onDifficultyChange(selectedDifficulty === diff ? null : diff)}
                className="px-3 py-1 rounded text-sm transition-all hover:scale-105"
                style={{
                  backgroundColor: selectedDifficulty === diff 
                    ? 'rgba(234, 179, 8, 0.2)' 
                    : 'var(--bg-secondary)',
                  color: selectedDifficulty === diff 
                    ? '#eab308' 
                    : 'var(--text-secondary)',
                  border: `1px solid ${
                    selectedDifficulty === diff 
                      ? 'rgba(234, 179, 8, 0.3)' 
                      : 'var(--border-color)'
                  }`
                }}
              >
                {'⭐'.repeat(diff)}
              </button>
            ))}
          </div>
        </div>

        {/* Result Count */}
        <div className="ml-auto">
          <span 
            className="text-sm"
            style={{ color: 'var(--text-tertiary)' }}
          >
            共 {resultCount} 题
          </span>
        </div>

        {/* Clear Filters */}
        {hasFilters && (
          <button
            onClick={() => {
              onSearchChange('');
              onHotChange(null);
              onDifficultyChange(null);
            }}
            className="text-sm px-3 py-1 rounded transition-all hover:opacity-70"
            style={{ 
              color: 'var(--accent-color)',
              backgroundColor: 'transparent'
            }}
          >
            清除筛选
          </button>
        )}
      </div>
    </div>
  );
}
