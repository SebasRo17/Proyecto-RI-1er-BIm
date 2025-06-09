import { SearchResult } from '../types/SearchResult';

// Simulated corpus of documents
const documentCorpus: { id: string; title: string; content: string }[] = [
  {
    id: 'doc1',
    title: 'Introducción a los Sistemas de Recuperación de Información',
    content: 'Los sistemas de recuperación de información (RI) son herramientas diseñadas para obtener información relevante a partir de una colección de recursos. Estos sistemas utilizan algoritmos para encontrar documentos que mejor coincidan con las consultas de los usuarios.',
  },
  {
    id: 'doc2',
    title: 'Modelos de Recuperación de Información',
    content: 'Existen diversos modelos de recuperación de información, como el modelo booleano, el modelo vectorial y el modelo probabilístico. Cada uno tiene sus propias características y ventajas dependiendo del contexto de aplicación.',
  },
  {
    id: 'doc3',
    title: 'Procesamiento del Lenguaje Natural en RI',
    content: 'El procesamiento del lenguaje natural (PLN) es fundamental en los sistemas de recuperación de información modernos. Técnicas como la lematización, la eliminación de palabras vacías y el análisis semántico mejoran significativamente los resultados.',
  },
  {
    id: 'doc4',
    title: 'Evaluación de Sistemas de RI',
    content: 'La evaluación de los sistemas de RI se realiza mediante métricas como precisión, exhaustividad, F1-score y MAP (Mean Average Precision). Estas métricas permiten comparar el rendimiento de diferentes algoritmos y sistemas.',
  },
  {
    id: 'doc5',
    title: 'Indexación en Sistemas de RI',
    content: 'La indexación es un proceso crucial en los sistemas de RI que permite acceder rápidamente a los documentos relevantes. Los índices invertidos son estructuras de datos comunes que asocian términos con documentos que los contienen.',
  },
  {
    id: 'doc6',
    title: 'Recuperación de Información en la Web',
    content: 'Los motores de búsqueda web son ejemplos prominentes de sistemas de RI. Utilizan técnicas avanzadas como el análisis de enlaces (PageRank) y la personalización para mejorar la relevancia de los resultados para los usuarios.',
  },
  {
    id: 'doc7',
    title: 'Sistemas de Recomendación',
    content: 'Los sistemas de recomendación son una aplicación especializada de RI que sugiere ítems basándose en preferencias del usuario. Utilizan filtrado colaborativo, filtrado basado en contenido o enfoques híbridos para generar recomendaciones personalizadas.',
  },
];

/**
 * Simulated search function that finds documents matching the query
 * @param query The search query string
 * @param topK Number of results to return (optional, defaults to 5)
 * @returns Array of search results
 */
export const buscarDocumentos = (query: string, topK: number = 5): SearchResult[] => {
  // Simple search implementation - convert to lowercase for case-insensitive search
  const normalizedQuery = query.toLowerCase().trim();
  
  if (!normalizedQuery) {
    return [];
  }

  // Search in title and content
  const results = documentCorpus
    .filter(doc => 
      doc.title.toLowerCase().includes(normalizedQuery) || 
      doc.content.toLowerCase().includes(normalizedQuery)
    )
    .map(doc => {
      // Find the relevant fragment containing the query
      let fragmento = doc.content;
      
      // If content is long, try to extract the relevant part
      if (doc.content.length > 150) {
        const index = doc.content.toLowerCase().indexOf(normalizedQuery);
        if (index >= 0) {
          // Extract text around the match
          const start = Math.max(0, index - 70);
          const end = Math.min(doc.content.length, index + normalizedQuery.length + 70);
          fragmento = (start > 0 ? '...' : '') + 
                      doc.content.substring(start, end) + 
                      (end < doc.content.length ? '...' : '');
        } else {
          // If query is in title but not in content, just take the first part
          fragmento = doc.content.substring(0, 150) + '...';
        }
      }
      
      return {
        id: doc.id,
        titulo: doc.title,
        fragmento
      };
    })
    .slice(0, topK);
    
  return results;
};