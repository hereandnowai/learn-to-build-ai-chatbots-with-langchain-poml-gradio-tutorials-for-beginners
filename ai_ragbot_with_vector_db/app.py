import gradio as gr
import os
import requests
import json
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # CORRECTED IMPORT
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# --- 1. Load Environment Variables & Initialize Models ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("🔴 GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file.")

# Initialize the LLM for conversation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# Initialize the local embedding model using the general class
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings( # CORRECTED CLASS NAME
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# --- 2. Create Vector Store with Caching ---
PDF_FILE_NAME = "About_HERE_AND_NOW_AI.pdf"
VECTOR_STORE_PATH = "faiss_index_bge" # Folder to save the FAISS index

if os.path.exists(VECTOR_STORE_PATH):
    # If cache exists, load it from the file system
    print("✅ Loading cached vector store...")
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    # If no cache, build the vector store from scratch
    print("🔵 No cache found. Building new vector store...")
    if not os.path.exists(PDF_FILE_NAME):
        print(f"Downloading PDF to {PDF_FILE_NAME}...")
        url = "https://raw.githubusercontent.com/hereandnowai/rag-workshop/main/pdfs/About_HERE_AND_NOW_AI.pdf"
        response = requests.get(url)
        with open(PDF_FILE_NAME, "wb") as f: f.write(response.content)

    # Load, split, and create embeddings
    loader = PyPDFLoader(PDF_FILE_NAME)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("Creating embeddings with local model and saving to cache. This might take a moment...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"✅ Vector store created and cached at '{VECTOR_STORE_PATH}'")

# Create the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- 3. Create the RAG Chain with Memory ---
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are Caramel AI, an expert assistant for HERE AND NOW AI.
Use the following pieces of retrieved context to answer the question.
If the information is not in the context, state that you cannot answer based on the provided information.

Context:
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(context=contextualized_question | retriever)
    | qa_prompt
    | llm
)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- 4. Gradio Interface ---
def ai_chatbot(message, history, session_id="main_session"):
    try:
        response = conversational_rag_chain.invoke(
            {"question": message},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content
    except Exception as e:
        print(f"🔴 Error during invocation: {e}")
        return "Sorry, an error occurred. Please check the terminal."

if __name__ == "__main__":
    gr.ChatInterface(
        fn=ai_chatbot,
        title="RAG from Vector DB",
        examples=[["Who is the CEO?"],["What courses are offered?"],["What is the contact info?"]],
        theme='monochrome'
    ).launch()