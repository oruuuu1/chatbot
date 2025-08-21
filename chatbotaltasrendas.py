import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2

# --- Função para ler PDF e agrupar em blocos ---
def ler_pdf_em_blocos(caminho_pdf, linhas_por_bloco=5):
    texto = ""
    try:
        with open(caminho_pdf, "rb") as arquivo:
            leitor = PyPDF2.PdfReader(arquivo)
            for pagina in leitor.pages:
                texto += pagina.extract_text() + "\n"
    except FileNotFoundError:
        st.error(f"Arquivo PDF não encontrado: {caminho_pdf}")
        return []

    # Quebrar em linhas e agrupar em blocos
    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    blocos = []
    for i in range(0, len(linhas), linhas_por_bloco):
        bloco = " ".join(linhas[i:i+linhas_por_bloco])
        if len(bloco) > 20:  # filtrar blocos muito curtos
            blocos.append(bloco)
    return blocos

# --- Carrega PDF e gera embeddings apenas uma vez ---
CAMINHO_PDF = "PL1087-RESUMO.pdf"
st.sidebar.info("Carregando PDF e embeddings, aguarde...")
blocos_texto = ler_pdf_em_blocos(CAMINHO_PDF)

# Modelo mais preciso
modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_blocos = modelo.encode(blocos_texto, convert_to_tensor=True)

# --- Interface Streamlit ---
st.title("📘 Chatbot - Projeto de Lei")
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    # Gera embedding da pergunta
    embedding_pergunta = modelo.encode(pergunta, convert_to_tensor=True)
    # Calcula similaridade
    similaridades = util.pytorch_cos_sim(embedding_pergunta, embeddings_blocos)[0]
    indice = torch.argmax(similaridades).item()
    resposta = blocos_texto[indice] if blocos_texto else "Não consegui encontrar a informação."
    st.write("**Resposta:**", resposta)
