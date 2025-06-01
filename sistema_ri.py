import os
import math
import nltk
import pickle
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SistemaRI:
    def __init__(self, idioma='spanish', usar_stemming=True, cache_dir='cache'):
        """
        Sistema de Recuperación de Información optimizado
        
        Args:
            idioma: Idioma para stopwords y stemming
            usar_stemming: Si usar stemming o no
            cache_dir: Directorio para cache de índices
        """
        self.idioma = idioma
        self.usar_stemming = usar_stemming
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Inicializar componentes de NLTK
        self._descargar_recursos_nltk()
        
        # Configurar herramientas de procesamiento
        self.stop_words = set(stopwords.words(idioma))
        self.stemmer = SnowballStemmer(idioma) if usar_stemming else None
        
        # Estructuras de datos del sistema
        self.corpus_crudo = []
        self.corpus_tokens = []
        self.nombres_archivos = []
        self.df = {}
        self.idf = {}
        self.tfs = []
        self.tfidfs = []
        self.bm25 = None
        self.vectorizer_sklearn = None
        self.matriz_tfidf_sklearn = None
        
    def _descargar_recursos_nltk(self):
        """Descarga recursos necesarios de NLTK"""
        recursos = ['punkt', 'stopwords']
        for recurso in recursos:
            try:
                nltk.data.find(f'tokenizers/{recurso}')
            except LookupError:
                logger.info(f"Descargando recurso NLTK: {recurso}")
                nltk.download(recurso, quiet=True)
    
    def limpiar_texto(self, texto: str) -> List[str]:
        """
        Limpia y tokeniza el texto
        
        Args:
            texto: Texto a limpiar
            
        Returns:
            Lista de tokens limpios
        """
        # Tokenización
        tokens = word_tokenize(texto.lower())
        
        # Filtrar tokens
        tokens_filtrados = []
        for token in tokens:
            # Mantener solo tokens alfanuméricos y no stopwords
            if token.isalnum() and token not in self.stop_words and len(token) > 2:
                # Aplicar stemming si está habilitado
                if self.stemmer:
                    token = self.stemmer.stem(token)
                tokens_filtrados.append(token)
        
        return tokens_filtrados
    
    def cargar_corpus(self, directorio: str, extensiones=('.txt', '.md', '.doc')):
        """
        Carga corpus desde directorio con cache inteligente
        
        Args:
            directorio: Directorio con documentos
            extensiones: Extensiones de archivo válidas
        """
        directorio = Path(directorio)
        cache_file = self.cache_dir / f"corpus_{directorio.name}.pkl"
        
        # Verificar si existe cache válido
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Verificar si el cache sigue siendo válido
                if self._cache_valido(cache_data, directorio):
                    logger.info("Cargando corpus desde cache")
                    self.corpus_crudo = cache_data['corpus_crudo']
                    self.nombres_archivos = cache_data['nombres_archivos']
                    self.corpus_tokens = cache_data['corpus_tokens']
                    return
            except Exception as e:
                logger.warning(f"Error al cargar cache: {e}")
        
        # Cargar corpus desde archivos
        logger.info(f"Cargando corpus desde {directorio}")
        self.corpus_crudo = []
        self.nombres_archivos = []
        
        archivos_procesados = 0
        for archivo in directorio.glob('**/*'):
            if archivo.suffix.lower() in extensiones and archivo.is_file():
                try:
                    with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
                        texto = f.read().strip()
                        if texto:  # Solo agregar archivos no vacíos
                            self.corpus_crudo.append(texto)
                            self.nombres_archivos.append(archivo.name)
                            archivos_procesados += 1
                except Exception as e:
                    logger.warning(f"Error al leer {archivo}: {e}")
        
        logger.info(f"Corpus cargado: {archivos_procesados} documentos")
        
        # Tokenizar corpus
        self.corpus_tokens = [self.limpiar_texto(doc) for doc in self.corpus_crudo]
        
        # Guardar en cache
        self._guardar_cache(cache_file, directorio)
    
    def _cache_valido(self, cache_data: dict, directorio: Path) -> bool:
        """Verifica si el cache sigue siendo válido"""
        try:
            cache_timestamp = cache_data.get('timestamp', 0)
            dir_timestamp = directorio.stat().st_mtime
            return cache_timestamp >= dir_timestamp
        except:
            return False
    
    def _guardar_cache(self, cache_file: Path, directorio: Path):
        """Guarda el corpus en cache"""
        try:
            cache_data = {
                'corpus_crudo': self.corpus_crudo,
                'nombres_archivos': self.nombres_archivos,
                'corpus_tokens': self.corpus_tokens,
                'timestamp': directorio.stat().st_mtime
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Cache guardado exitosamente")
        except Exception as e:
            logger.warning(f"Error al guardar cache: {e}")
    
    def calcular_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calcula Term Frequency de manera optimizada"""
        if not tokens:
            return {}
        
        # Usar Counter para eficiencia
        tf_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Normalizar por total de tokens
        return {palabra: count / total_tokens for palabra, count in tf_counts.items()}
    
    def calcular_df_idf(self):
        """Calcula Document Frequency e Inverse Document Frequency"""
        if not self.corpus_tokens:
            raise ValueError("Corpus no cargado")
        
        logger.info("Calculando DF e IDF")
        
        # Calcular DF usando defaultdict para eficiencia
        self.df = defaultdict(int)
        N = len(self.corpus_tokens)
        
        for doc_tokens in self.corpus_tokens:
            # Usar set para evitar contar palabras repetidas en el mismo documento
            for palabra in set(doc_tokens):
                self.df[palabra] += 1
        
        # Calcular IDF
        self.idf = {
            palabra: math.log(N / df_count) 
            for palabra, df_count in self.df.items()
        }
        
        logger.info(f"Vocabulario: {len(self.df)} términos únicos")
    
    def calcular_tfidf_manual(self):
        """Calcula TF-IDF manualmente para todos los documentos"""
        logger.info("Calculando TF-IDF manual")
        
        # Calcular TF para todos los documentos
        self.tfs = [self.calcular_tf(doc_tokens) for doc_tokens in self.corpus_tokens]
        
        # Calcular TF-IDF para todos los documentos
        self.tfidfs = []
        for tf in self.tfs:
            tfidf = {
                palabra: tf_val * self.idf.get(palabra, 0)
                for palabra, tf_val in tf.items()
            }
            self.tfidfs.append(tfidf)
    
    def inicializar_bm25(self, k1=1.5, b=0.75):
        """Inicializa BM25 con parámetros optimizados"""
        if not self.corpus_tokens:
            raise ValueError("Corpus no cargado")
        
        logger.info("Inicializando BM25")
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)
    
    def inicializar_sklearn_tfidf(self, max_features=10000):
        """Inicializa TF-IDF usando sklearn para comparación"""
        logger.info("Inicializando TF-IDF de sklearn")
        
        # Reconstruir documentos como strings para sklearn
        documentos_string = [' '.join(tokens) for tokens in self.corpus_tokens]
        
        self.vectorizer_sklearn = TfidfVectorizer(
            max_features=max_features,
            lowercase=False,  # Ya está procesado
            token_pattern=r'\b\w+\b'
        )
        
        self.matriz_tfidf_sklearn = self.vectorizer_sklearn.fit_transform(documentos_string)
    
    def construir_indices(self, directorio: str):
        """Construye todos los índices del sistema"""
        logger.info("Construyendo índices del sistema RI")
        
        # Cargar corpus
        self.cargar_corpus(directorio)
        
        if not self.corpus_tokens:
            raise ValueError("No se encontraron documentos válidos en el directorio")
        
        # Calcular métricas
        self.calcular_df_idf()
        self.calcular_tfidf_manual()
        self.inicializar_bm25()
        self.inicializar_sklearn_tfidf()
        
        logger.info("Sistema RI inicializado correctamente")
    
    def buscar_bm25(self, consulta: str, top_k=10) -> List[Tuple[str, float]]:
        """Búsqueda usando BM25"""
        if not self.bm25:
            raise ValueError("BM25 no inicializado")
        
        consulta_tokens = self.limpiar_texto(consulta)
        if not consulta_tokens:
            return []
        
        scores = self.bm25.get_scores(consulta_tokens)
        resultados = list(zip(self.nombres_archivos, scores))
        
        # Ordenar por score descendente y retornar top_k
        return sorted(resultados, key=lambda x: x[1], reverse=True)[:top_k]
    
    def buscar_tfidf_manual(self, consulta: str, top_k=10) -> List[Tuple[str, float]]:
        """Búsqueda usando TF-IDF manual (similaridad coseno)"""
        consulta_tokens = self.limpiar_texto(consulta)
        if not consulta_tokens:
            return []
        
        # Calcular TF-IDF de la consulta
        consulta_tf = self.calcular_tf(consulta_tokens)
        consulta_tfidf = {
            palabra: tf_val * self.idf.get(palabra, 0)
            for palabra, tf_val in consulta_tf.items()
        }
        
        # Calcular similaridad coseno con cada documento
        scores = []
        for i, doc_tfidf in enumerate(self.tfidfs):
            score = self._similaridad_coseno(consulta_tfidf, doc_tfidf)
            scores.append((self.nombres_archivos[i], score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _similaridad_coseno(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calcula similaridad coseno entre dos vectores TF-IDF"""
        # Intersección de palabras
        palabras_comunes = set(vec1.keys()) & set(vec2.keys())
        
        if not palabras_comunes:
            return 0.0
        
        # Producto punto
        producto_punto = sum(vec1[palabra] * vec2[palabra] for palabra in palabras_comunes)
        
        # Normas
        norma1 = math.sqrt(sum(val**2 for val in vec1.values()))
        norma2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if norma1 == 0 or norma2 == 0:
            return 0.0
        
        return producto_punto / (norma1 * norma2)
    
    def buscar_combinado(self, consulta: str, top_k=10, peso_bm25=0.7, peso_tfidf=0.3) -> List[Tuple[str, float]]:
        """Búsqueda combinando BM25 y TF-IDF"""
        resultados_bm25 = self.buscar_bm25(consulta, len(self.nombres_archivos))
        resultados_tfidf = self.buscar_tfidf_manual(consulta, len(self.nombres_archivos))
        
        # Normalizar scores
        max_bm25 = max(score for _, score in resultados_bm25) if resultados_bm25 else 1
        max_tfidf = max(score for _, score in resultados_tfidf) if resultados_tfidf else 1
        
        # Crear diccionarios para facilitar búsqueda
        scores_bm25 = {nombre: score/max_bm25 for nombre, score in resultados_bm25}
        scores_tfidf = {nombre: score/max_tfidf for nombre, score in resultados_tfidf}
        
        # Combinar scores
        scores_combinados = []
        for nombre in self.nombres_archivos:
            score_bm25 = scores_bm25.get(nombre, 0)
            score_tfidf = scores_tfidf.get(nombre, 0)
            score_final = peso_bm25 * score_bm25 + peso_tfidf * score_tfidf
            scores_combinados.append((nombre, score_final))
        
        return sorted(scores_combinados, key=lambda x: x[1], reverse=True)[:top_k]
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas del corpus"""
        if not self.corpus_tokens:
            print("❌ Corpus no cargado")
            return
        
        total_tokens = sum(len(doc) for doc in self.corpus_tokens)
        vocabulario_size = len(self.df)
        promedio_tokens = total_tokens / len(self.corpus_tokens)
        
        print(f"\n📊 Estadísticas del Corpus:")
        print(f"   • Documentos: {len(self.corpus_tokens)}")
        print(f"   • Total de tokens: {total_tokens:,}")
        print(f"   • Vocabulario único: {vocabulario_size:,}")
        print(f"   • Promedio tokens/doc: {promedio_tokens:.1f}")
        
        # Mostrar términos más frecuentes
        if self.df:
            terminos_frecuentes = sorted(self.df.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n🏆 Términos más frecuentes:")
            for termino, freq in terminos_frecuentes:
                print(f"   • {termino}: {freq} documentos")
    
    def consulta_interactiva(self):
        """Interfaz interactiva para consultas"""
        print("\n🔍 Sistema de Recuperación de Información")
        print("=" * 50)
        
        while True:
            try:
                consulta = input("\n➤ Ingresa tu consulta (o 'salir' para terminar): ").strip()
                
                if consulta.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if not consulta:
                    continue
                
                print(f"\n🔎 Buscando: '{consulta}'")
                print("-" * 40)
                
                # Realizar búsquedas
                resultados_bm25 = self.buscar_bm25(consulta, 5)
                resultados_tfidf = self.buscar_tfidf_manual(consulta, 5)
                resultados_combinados = self.buscar_combinado(consulta, 5)
                
                # Mostrar resultados
                print("\n📄 Resultados BM25:")
                for i, (nombre, score) in enumerate(resultados_bm25, 1):
                    print(f"   {i}. {nombre} (Score: {score:.4f})")
                
                print("\n📄 Resultados TF-IDF:")
                for i, (nombre, score) in enumerate(resultados_tfidf, 1):
                    print(f"   {i}. {nombre} (Score: {score:.4f})")
                
                print("\n📄 Resultados Combinados:")
                for i, (nombre, score) in enumerate(resultados_combinados, 1):
                    print(f"   {i}. {nombre} (Score: {score:.4f})")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Función principal"""
    # Configurar sistema
    sistema = SistemaRI(idioma='spanish', usar_stemming=True)
    
    try:
        # Directorio del corpus
        directorio_corpus = input("📁 Ingresa el directorio del corpus (default: 'corpus'): ").strip() or 'corpus'
        
        if not Path(directorio_corpus).exists():
            print(f"❌ El directorio '{directorio_corpus}' no existe")
            return
        
        # Construir índices
        sistema.construir_indices(directorio_corpus)
        
        # Mostrar estadísticas
        sistema.mostrar_estadisticas()
        
        # Modo interactivo
        sistema.consulta_interactiva()
        
    except Exception as e:
        logger.error(f"Error en main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()