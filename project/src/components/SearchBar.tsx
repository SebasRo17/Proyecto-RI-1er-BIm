import React, { useState } from 'react';
import { Search } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string, topK: number) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query, topK);
  };

  return (
    <form onSubmit={handleSearch} className="w-full max-w-3xl mx-auto mb-8">
      <div className="flex flex-col md:flex-row gap-4">
        <div className="relative flex-grow">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Escribe tu consulta..."
            className="w-full p-3 pr-10 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            required
          />
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-gray-400">
            <Search size={20} />
          </div>
        </div>
        
        <div className="flex gap-2">
          <div className="w-32">
            <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-1">
              Resultados (top K)
            </label>
            <input
              id="topK"
              type="number"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
              className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            />
          </div>
          
          <button
            type="submit"
            className="self-end px-6 py-3 bg-blue-600 text-white font-medium rounded-lg shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all"
          >
            Buscar
          </button>
        </div>
      </div>
    </form>
  );
};

export default SearchBar;