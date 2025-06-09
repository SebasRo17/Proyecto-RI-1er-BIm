import { SearchResult } from '../types/SearchResult';

/**
 * Consulta la API de búsqueda y devuelve los resultados sin procesar localmente.
 * Se espera que el backend devuelva objetos con al menos `doc_id` y `score`,
 * y opcionalmente `snippet` si así se ha implementado.
 * 
 * @param query Consulta de búsqueda
 * @param topK Número máximo de resultados
 * @returns Resultados directamente desde la API
 */
export const buscarDocumentos = async (
  query: string,
  topK: number = 10
): Promise<SearchResult[]> => {
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

    // Mapear los resultados a la interfaz SearchResult
    const resultados: SearchResult[] = data.results.map(
      (item: { doc_id: string; score: number; snippet?: string }) => ({
        id: item.doc_id,
        titulo: `Documento ${item.doc_id}`,
        fragmento: `Score: ${item.score.toFixed(2)}`,
        snippet: item.snippet // puede ser undefined o string, está bien
      })
    );

    return resultados;
  } catch (error) {
    console.error('Error en la búsqueda:', error);
    return [];
  }
};
