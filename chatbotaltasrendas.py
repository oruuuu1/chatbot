from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2

app = Flask(__name__)

# Função para ler PDF
def ler_pdf(caminho_pdf):
    texto = ""
    with open(caminho_pdf, "rb") as arquivo:
        leitor = PyPDF2.PdfReader(arquivo)
        for pagina in leitor.pages:
            texto += pagina.extract_text() + "\n"
    return texto

# Caminho do PDF do projeto de lei
CAMINHO_PDF = r"C:/Users/oruam/Downloads/projeto_lei.pdf.pdf"  # coloque o arquivo na mesma pasta do código

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
    resposta = paragrafos[indice]

    return jsonify({"fulfillmentText": resposta})

if __name__ == "__main__":
    app.run(port=5000, debug=True)

