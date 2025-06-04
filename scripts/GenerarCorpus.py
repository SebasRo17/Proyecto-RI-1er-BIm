import os
import ir_datasets

# Cargar el dataset de consultas
dataset = ir_datasets.load("car/v1.5/test200")

# Definir carpeta de salida
output_dir = r"D:\Universidad\8 - Octavo\Recuperacion de la informacion\Proyecto-RI-1er-BIm\corpus"
os.makedirs(output_dir, exist_ok=True)

# Guardar cada consulta como un archivo .txt
count = 0
for query in dataset.queries_iter():
    query_text = query.text.strip()
    query_id = query.query_id.replace("/", "_")  # Por si hay caracteres no válidos
    file_path = os.path.join(output_dir, f"query_{query_id}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(query_text)
        count += 1

print(f"{count} consultas guardadas en la carpeta '{output_dir}'")
