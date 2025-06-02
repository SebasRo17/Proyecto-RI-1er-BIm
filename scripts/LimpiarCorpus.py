import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

# Descargar recursos necesarios
nltk.download('stopwords')

# Inicializar tokenizer manualmente sin usar sent_tokenize
tokenizer = TreebankWordTokenizer()

# Rutas del corpus
input_dir = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus"
output_dir = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus_limpio"
os.makedirs(output_dir, exist_ok=True)

# Stopwords
stop_words = set(stopwords.words('english'))

def limpiar_texto(texto):
    # Minúsculas y eliminación de puntuación
    texto = texto.lower().translate(str.maketrans("", "", string.punctuation))
    # Tokenizar usando TreebankTokenizer (no usa punkt)
    tokens = tokenizer.tokenize(texto)
    # Filtrar solo palabras alfabéticas que no sean stopwords
    tokens_limpios = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens_limpios)

# Procesar documentos
for archivo in os.listdir(input_dir):
    if archivo.endswith(".txt"):
        ruta_entrada = os.path.join(input_dir, archivo)
        ruta_salida = os.path.join(output_dir, archivo)

        with open(ruta_entrada, "r", encoding="utf-8") as f_in:
            texto = f_in.read()
            texto_limpio = limpiar_texto(texto)

        with open(ruta_salida, "w", encoding="utf-8") as f_out:
            f_out.write(texto_limpio)

print(f"Corpus limpiado y guardado en: {output_dir}")
