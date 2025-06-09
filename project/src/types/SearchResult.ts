export interface SearchResult {
  doc_id: string;
  titulo: string;
  score: number;
  snippet?: string;
  full_text?: string;
}