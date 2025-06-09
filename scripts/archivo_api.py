from fastapi.responses import PlainTextResponse

CORPUS_ORIGINAL_DIR = r"C:\Users\roble\OneDrive\Documentos\GitHub\Proyecto-RI-1er-BIm\corpus2"

@app.get("/document/{doc_name}", response_class=PlainTextResponse, summary="Obtener documento original por nombre")
def get_document(doc_name: str):
    # Siempre quita el sufijo '_clean'
    base_name = doc_name.replace("_clean", "")
    fname = f"{base_name}.txt"
    path = os.path.join(CORPUS_ORIGINAL_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Documento '{doc_name}' no encontrado en corpus original")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return content

