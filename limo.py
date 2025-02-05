import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import re

# Folder containing pre-uploaded PDFs
PDF_FOLDER = "pdfs"

# Initialize Streamlit app
st.set_page_config(page_icon="💬", layout="wide", page_title="Groq PDF Chatbot")

# Sidebar for model selection
with st.sidebar:
    st.title('🤗💬 LiMO')
    st.write("**Select Model & Settings**")
    
    models = {
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    }
    
    model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0)
    max_tokens = st.slider("Max Tokens:", min_value=512, max_value=models[model_option]["tokens"], value=2048, step=512)

# Display icon and title
st.subheader("LiMO, Light finance's helper", divider="rainbow", anchor=False)

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Load and process PDFs (only once)
if st.session_state.vectorstore is None:
    docs = []
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            loader = PDFPlumberLoader(os.path.join(PDF_FOLDER, file))
            docs.extend(loader.load())
    
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)
    embedder = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embedder)
    st.session_state.vectorstore = vectorstore

retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar='🤖' if msg["role"] == "assistant" else '👨‍💻'):
        st.markdown(msg["content"])

# Function to clean response
def clean_response(response):
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# Chat input
if prompt := st.chat_input("Ask about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
    
    # Retrieve relevant PDF context
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = f"Use only the following context to answer the question: {context}\nQuestion: {prompt}\nAnswer:"
    
    # Fetch response from Groq API
    try:
        response = client.chat.completions.create(
            model=model_option,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=max_tokens
        )
        bot_response = clean_response(response.choices[0].message.content)
    except Exception as e:
        bot_response = f"Error: {e}"
    
    # Display response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_response)
