import ir_datasets
import os

# Ruta de salida
output_dir = r"C:\Users\roble\OneDrive\Documentos\GitHub\Proyecto-RI-1er-BIm\corpus2"
os.makedirs(output_dir, exist_ok=True)

dataset = ir_datasets.load("beir/cqadupstack/mathematica")

for doc in dataset.docs_iter():
    doc_id = doc.doc_id.replace('/', '_').replace('\\', '_')
    if len(doc_id) > 100:
        doc_id = doc_id[:100]
    filename = os.path.join(output_dir, f"{doc_id}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(doc.text)
    except Exception as e:
        print(f"Error en {doc_id}: {e}")

print("Descarga completada.")
