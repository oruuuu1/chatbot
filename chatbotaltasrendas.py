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

    linhas = [l.strip() for l in texto.split("\n") if l.strip()]
    blocos = []
    for i in range(0, len(linhas), linhas_por_bloco):
        bloco = " ".join(linhas[i:i+linhas_por_bloco])
        if len(bloco) > 20:  # filtra blocos muito curtos
            blocos.append(bloco)
    return blocos

# --- Carrega PDF e gera embeddings ---
CAMINHO_PDF = "PL1087-RESUMO.pdf"
st.sidebar.info("Carregando PDF e embeddings, aguarde...")
blocos_texto = ler_pdf_em_blocos(CAMINHO_PDF)

modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_blocos = modelo.encode(blocos_texto, convert_to_tensor=True)

# --- Interface Streamlit ---
st.title("📘 Chatbot baseado em PDF - Projeto de Lei")
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    embedding_pergunta = modelo.encode(pergunta, convert_to_tensor=True)
    similaridades = util.pytorch_cos_sim(embedding_pergunta, embeddings_blocos)[0]

    # Seleciona todos os blocos com similaridade maior que um limiar (ex: 0.5)
    limiar = 0.5
    blocos_relevantes = [blocos_texto[i] for i, sim in enumerate(similaridades) if sim > limiar]

    if blocos_relevantes:
        resposta = "\n\n".join(blocos_relevantes)
    else:
        # Se nada passar do limiar, pega o bloco mais similar
        indice = torch.argmax(similaridades).item()
        resposta = blocos_texto[indice]

    st.write("**Resposta baseada no PDF:**")
    st.write(resposta)
