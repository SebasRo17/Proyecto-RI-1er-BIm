import React from 'react';
import { SearchResult } from '../types/SearchResult';

interface SearchResultsProps {
  results: SearchResult[];       // Asegúrate de que esto siempre sea un array
  isLoading: boolean;
  hasSearched: boolean;
}

const SearchResults: React.FC<SearchResultsProps> = ({ results, isLoading, hasSearched }) => {
  // Validación defensiva para evitar errores si 'results' no es un arreglo
  const safeResults: SearchResult[] = Array.isArray(results) ? results : [];

  if (isLoading) {
    return (
      <div className="w-full max-w-3xl mx-auto p-8 bg-white rounded-lg shadow-md animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-3/4 mb-6"></div>
        <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
        <div className="h-3 bg-gray-200 rounded w-5/6 mb-2"></div>
        <div className="h-3 bg-gray-200 rounded w-4/6 mb-8"></div>

        <div className="h-4 bg-gray-200 rounded w-2/3 mb-6"></div>
        <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
        <div className="h-3 bg-gray-200 rounded w-5/6 mb-2"></div>
        <div className="h-3 bg-gray-200 rounded w-4/6"></div>
      </div>
    );
  }

  if (hasSearched && safeResults.length === 0) {
    return (
      <div className="w-full max-w-3xl mx-auto p-8 bg-white rounded-lg shadow-md text-center">
        <p className="text-gray-600">No se encontraron resultados para tu búsqueda.</p>
        <p className="text-gray-500 mt-2 text-sm">Intenta con términos más generales o verifica la ortografía.</p>
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="w-full max-w-3xl mx-auto p-8 bg-white rounded-lg shadow-md text-center">
        <p className="text-gray-600">Ingresa una consulta y haz clic en "Buscar" para ver resultados.</p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Resultados ({safeResults.length})</h2>

      <div className="space-y-6">
        {safeResults.map((result) => (
          <div
            key={result.id}
            className="p-6 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow"
          >
            <h3 className="text-lg font-semibold text-blue-700 mb-2">{result.titulo}</h3>
            <p className="text-gray-700">{result.fragmento}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
