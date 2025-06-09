import { SearchResult } from '../types/SearchResult';

/**
 * Consulta la API de búsqueda y devuelve los resultados directamente tipados como SearchResult.
 * 
 * @param query Consulta de búsqueda
 * @param topK Número máximo de resultados
 * @returns Resultados directamente desde la API
 */
export const buscarDocumentos = async (query: string, topK: number = 10): Promise<SearchResult[]> => {
  const endpoint = `http://127.0.0.1:8000/search?query=${encodeURIComponent(query)}&topk=${topK}`;

  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: {
        'accept': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Error al consultar la API: ${response.statusText}`);
    }

    const data = await response.json();
    console.log("Datos recibidos de la API:", data);  

    // Retornar directamente los resultados si coinciden con la estructura esperada
    const resultados: SearchResult[] = data.results.map((item: any) => ({
      doc_id: item.doc_id,
      score: item.score,
      snippet: item.snippet || '',        // si no hay snippet, poner cadena vacía
      full_text: item.full_text || ''     // opcionalmente incluir el texto completo
    }));

    return resultados;
  } catch (error) {
    console.error('Error en la búsqueda:', error);
    return [];
  }
};
