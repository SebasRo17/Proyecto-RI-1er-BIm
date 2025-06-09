import React, { useState } from 'react';
import { SearchResult } from '../types/SearchResult';

interface SearchResultsProps {
  results: SearchResult[];
  isLoading: boolean;
  hasSearched: boolean;
}

const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  isLoading,
  hasSearched
}) => {
  // Modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [modalContent, setModalContent] = useState('');
  const [modalTitle, setModalTitle] = useState('');

  // Validación defensiva
  const safeResults: SearchResult[] = Array.isArray(results) ? results : [];

  // Handler para abrir el modal y pedir el documento original
  const handleOpenModal = async (docId: string) => {
    const cleanId = docId.replace('_clean', '');
    try {
      const response = await fetch(`http://127.0.0.1:8000/document/${cleanId}`);
      if (!response.ok) throw new Error('No se pudo obtener el documento');
      const text = await response.text();
      setModalContent(text);
      setModalTitle(`Documento ${cleanId}`);
      setModalOpen(true);
    } catch (error) {
      setModalContent('Error al cargar el documento.');
      setModalTitle('Error');
      setModalOpen(true);
    }
  };

  const handleCloseModal = () => setModalOpen(false);

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
      {/* Modal */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full p-6 relative">
            <button
              onClick={handleCloseModal}
              className="absolute top-2 right-2 text-gray-400 hover:text-red-500 text-xl font-bold"
              aria-label="Cerrar"
            >
              &times;
            </button>
            <h3 className="text-lg font-semibold mb-4 text-blue-700">{modalTitle}</h3>
            <pre className="max-h-[60vh] overflow-y-auto bg-gray-50 p-4 rounded text-gray-800 whitespace-pre-wrap">
              {modalContent}
            </pre>
          </div>
        </div>
      )}

      <h2 className="text-xl font-semibold mb-4 text-gray-800">Resultados ({safeResults.length})</h2>
      <div className="space-y-6">
        {safeResults.map((result) => (
          <div
            key={result.id}
            className="p-6 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow cursor-pointer"
            onClick={() => handleOpenModal(result.id)}
            title="Ver documento original"
          >
            <h3 className="text-lg font-semibold text-blue-700 mb-2">{result.titulo}</h3>
            <p className="text-gray-700">{result.fragmento}</p>
            {result.snippet && (
              <pre className="bg-gray-100 text-gray-800 rounded p-2 mt-2 text-sm whitespace-pre-wrap">
                {result.snippet}
              </pre>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
