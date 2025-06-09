export interface SearchResult {
  id: string;
  titulo: string;
  fragmento: string; // puedes renombrar si quieres, pero aquí lo mantenemos para compatibilidad
  snippet?: string;  // ← ¡esto es nuevo!
}
