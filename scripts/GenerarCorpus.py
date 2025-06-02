import os
from sklearn.datasets import fetch_20newsgroups

# Cargar el dataset sin encabezados, pies ni citas
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data

# Ruta relativa del directorio de salida
output_dir = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus"
os.makedirs(output_dir, exist_ok=True)

# Guardar cada documento como archivo de texto
for i, doc in enumerate(documents):
    file_path = os.path.join(output_dir, f"doc_{i+1}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(doc)

print(f"{len(documents)} documentos guardados en la carpeta '{output_dir}'")
