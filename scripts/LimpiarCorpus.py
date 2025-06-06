#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocesamiento_corpus.py

Recorre todos los archivos .txt en la carpeta INPUT_DIR,
les aplica limpieza (remueve URLs, caracteres especiales, puntuación, múltiples espacios),
tokenización, remoción de stopwords en inglés, lematización
y genera un archivo "limpio" por cada documento en OUTPUT_DIR.

Requisitos:
    pip install nltk
    (y haber descargado los recursos NLTK al menos una vez)

Uso:
    python preprocesamiento_corpus.py
"""

import os
import re
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize

# -------------- CONFIGURACIÓN (ajusta estas rutas si es necesario) --------------
INPUT_DIR = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus2"
OUTPUT_DIR = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus2_clean"

# Patrón regex para tokenizar solo secuencias de letras (en minúscula).
REGEX_PATTERN = r"[a-z]+"

# Idioma de stopwords: ingés (tu corpus está en inglés)
STOPWORD_LANG = "english"
# ----------------------------------------------------------------------------------


def inicializar_nltk():
    """
    Descarga los recursos NLTK necesarios (punkt, stopwords, wordnet).
    Solo descarga si no están disponibles.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


def limpiar_texto(texto: str) -> str:
    """
    Aplica limpieza “cruda” sobre el texto completo:
      1. Remueve URLs (http://... o https://...).
      2. Remueve cualquier carácter que no sea letra o espacio (es decir, puntuación, números, símbolos).
      3. Convierte a minúsculas.
      4. Colapsa múltiples espacios en uno solo y strip() inicial/final.
    Devuelve el texto “limpio” aún sin tokenizar.
    """
    # 1. Eliminar URLs (http(s)://...)
    texto_sin_urls = re.sub(r"http\S+", " ", texto)

    # 2. Reemplazar cualquier cosa que no sea letra o espacio por espacio
    texto_sin_puntuacion = re.sub(r"[^A-Za-z\s]", " ", texto_sin_urls)

    # 3. Pasa a minúsculas
    texto_min = texto_sin_puntuacion.lower()

    # 4. Colapsar múltiples espacios en uno y quitar espacios iniciales/finales
    texto_colapsado = re.sub(r"\s+", " ", texto_min).strip()

    return texto_colapsado


def procesar_texto(texto: str,
                   stopwords_set: set,
                   lemmatizer: WordNetLemmatizer) -> str:
    """
    Dado un texto (ya “limpio” en su forma cruda), hace:
      a) Tokenizar usando regexp_tokenize con el patrón REGEX_PATTERN
      b) Remover tokens vacíos o espacios residuales
      c) Remover stopwords en inglés
      d) Lematizar cada token restante
      e) Reconstruir cadena final (lemmas separados por espacio)
    Devuelve el texto preprocesado listo.
    """
    # a) Tokenización con regex (solo letras minúsculas)
    tokens = regexp_tokenize(texto, pattern=REGEX_PATTERN)

    # b) Filtrar tokens vacíos (por si acaso)
    tokens = [t for t in tokens if t.strip()]

    # c) Eliminar stopwords
    tokens_sin_sw = [t for t in tokens if t not in stopwords_set]

    # d) Lematizar cada token
    lemmas = [lemmatizer.lemmatize(t) for t in tokens_sin_sw]

    # e) Reconstruir texto final
    texto_prepro = " ".join(lemmas)
    return texto_prepro


def main(input_dir: str, output_dir: str):
    # 0. Inicializar recursos de NLTK
    inicializar_nltk()

    # 1. Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # 2. Construir el set de stopwords y el lemmatizer
    sw_set = set(stopwords.words(STOPWORD_LANG))
    lemmatizer = WordNetLemmatizer()

    # 3. Listar todos los archivos .txt en input_dir
    archivos_txt = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    total = len(archivos_txt)
    print(f"Se encontraron {total} archivos .txt en '{input_dir}'.\nIniciando preprocesamiento...")

    # 4. Procesar cada archivo uno a uno
    for idx, nombre_archivo in enumerate(archivos_txt, start=1):
        ruta_in = os.path.join(input_dir, nombre_archivo)

        # Leer el texto original
        with open(ruta_in, "r", encoding="utf-8") as f:
            texto_original = f.read()

        # 4.1. Limpieza cruda: borrar URLs, puntuación, caracteres especiales, múltiples espacios…
        texto_limpio_crudo = limpiar_texto(texto_original)

        # 4.2. Tokenizar, quitar stopwords, lematizar
        texto_preprocesado = procesar_texto(texto_limpio_crudo, sw_set, lemmatizer)

        # 4.3. Guardar resultado en carpeta de salida
        nombre_base, _ = os.path.splitext(nombre_archivo)
        nombre_salida = f"{nombre_base}_clean.txt"
        ruta_out = os.path.join(output_dir, nombre_salida)
        with open(ruta_out, "w", encoding="utf-8") as f_out:
            f_out.write(texto_preprocesado)

        # Mostrar progreso cada 500 archivos o al final
        if idx % 500 == 0 or idx == total:
            porcentaje = (idx / total) * 100
            print(f"  Procesados {idx}/{total} archivos ({porcentaje:.1f}%).")

    print("\nPreprocesamiento completado. Archivos limpios en:")
    print(f"  → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de corpus: limpieza pesada, tokenización, stopwords, lematización."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=INPUT_DIR,
        help="Carpeta donde están los .txt sin procesar.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Carpeta donde se guardarán los .txt preprocesados.",
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
