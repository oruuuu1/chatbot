from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2
import os

app = Flask(__name__)

# Função para ler PDF
def ler_pdf(caminho_pdf):
    texto = ""
    try:
        with open(caminho_pdf, "rb") as arquivo:
            leitor = PyPDF2.PdfReader(arquivo)
            for pagina in leitor.pages:
                texto += pagina.extract_text() + "\n"
    except FileNotFoundError:
        print(f"Arquivo PDF não encontrado: {caminho_pdf}")
        texto = "PDF não encontrado. Por favor, confira o arquivo."
    return texto

# Caminho do PDF (na mesma pasta do código)
CAMINHO_PDF = "projeto_lei.pdf"

# Lê e processa o texto do PDF
texto_lei = ler_pdf(CAMINHO_PDF)
paragrafos = [p.strip() for p in texto_lei.split("\n") if p.strip()]

# Modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_texto = modelo.encode(paragrafos, convert_to_tensor=True)

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json(force=True)
    pergunta = req.get("queryResult", {}).get("queryText", "")

    embedding_pergunta = modelo.encode(pergunta, convert_to_tensor=True)
    similaridades = util.pytorch_cos_sim(embedding_pergunta, embeddings_texto)[0]
    indice = torch.argmax(similaridades).item()
    resposta = paragrafos[indice] if paragrafos else "Não consegui encontrar a informação."

    return jsonify({"fulfillmentText": resposta})

if __name__ == "__main__":
    # Usa a porta fornecida pelo Render ou 5000 como fallback
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
