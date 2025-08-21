import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2

# Função para ler PDF
def ler_pdf(caminho_pdf):
    texto = ""
    with open(caminho_pdf, "rb") as arquivo:
        leitor = PyPDF2.PdfReader(arquivo)
        for pagina in leitor.pages:
            texto += pagina.extract_text() + "\n"
    return texto

# Lê o PDF
CAMINHO_PDF = "projeto_lei.pdf"
texto_lei = ler_pdf(CAMINHO_PDF)
paragrafos = [p.strip() for p in texto_lei.split("\n") if p.strip()]

# Modelo
modelo = SentenceTransformer("paraphrase-MiniLM-L3-v2")
embeddings_texto = modelo.encode(paragrafos, convert_to_tensor=True)

# Interface Streamlit
st.title("📘 Chatbot - Projeto de Lei")
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    embedding_pergunta = modelo.encode(pergunta, convert_to_tensor=True)
    similaridades = util.pytorch_cos_sim(embedding_pergunta, embeddings_texto)[0]
    indice = torch.argmax(similaridades).item()
    resposta = paragrafos[indice] if paragrafos else "Não consegui encontrar a informação."
    st.write("**Resposta:**", resposta)
