import React, { useState } from 'react';
import { Database } from 'lucide-react';
import SearchBar from './components/SearchBar';
import SearchResults from './components/SearchResults';
import { buscarDocumentos } from './services/searchService';
import { SearchResult } from './types/SearchResult';

function App() {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = (query: string, topK: number) => {
    setIsLoading(true);
    setHasSearched(true);
    
    // Simulate network delay
    setTimeout(() => {
      const searchResults = buscarDocumentos(query, topK);
      setResults(searchResults);
      setIsLoading(false);
    }, 800);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm py-4">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center">
            <Database className="h-6 w-6 text-blue-600 mr-2" />
            <h1 className="text-xl font-semibold text-gray-800">Sistema de Recuperación de Información</h1>
          </div>
        </div>
      </header>
      
      {/* Main content */}
      <main className="flex-grow py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto">
          <div className="mb-8 text-center">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Búsqueda de Documentos</h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Ingresa tu consulta para buscar en nuestro corpus de documentos. Puedes especificar el número de resultados a mostrar.
            </p>
          </div>
          
          <SearchBar onSearch={handleSearch} />
          
          <SearchResults 
            results={results} 
            isLoading={isLoading} 
            hasSearched={hasSearched} 
          />
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-white py-4 border-t border-gray-200">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500 text-sm">
            Sistema de Recuperación de Información - {new Date().getFullYear()}
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;