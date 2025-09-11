import gradio as gr
import os
import requests
import json
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# --- 1. Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("🔴 GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file.")

# --- 2. Download and Load the Knowledge Base into a String ---
file_path = "profile-of-hereandnowai.txt"

# Download the file if it doesn't exist locally
if not os.path.exists(file_path):
    print(f"Downloading knowledge base to {file_path}...")
    url = "https://raw.githubusercontent.com/hereandnowai/vac/refs/heads/master/prospectus-context.txt"
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)

# Read the entire file content into a single string variable
try:
    with open(file_path, "r", encoding="utf-8") as file:
        text_context = file.read()
    print("✅ Successfully loaded the text context.")
except Exception as e:
    print(f"🔴 Error reading text file: {e}")
    text_context = "Error: Could not read the context file."


# --- 3. Create the LangChain RAGbot (Context-Stuffing Method) ---
# Use the model from your original project_3/app.py file
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# This prompt template includes the ENTIRE text context for every query
# This is the same method your original script used
qa_system_prompt = f"""You are Caramel AI, an AI assistant for HERE AND NOW AI.
Answer the user's questions based only on the following context. Keep your answers concise.

Context:
---
{text_context}
---
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Create the chain without a retriever
rag_chain = qa_prompt | llm

# --- 4. Add Message History ---
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

# --- 5. Gradio Interface ---
def ai_chatbot(message, history, session_id="main_session"):
    try:
        response = conversational_rag_chain.invoke(
            {"question": message},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content
    except Exception as e:
        print(f"🔴 Error during invocation: {e}")
        return "Sorry, I encountered an error. Please check the terminal."

# Load branding data
with open('branding.json') as f:
    brand_info = json.load(f)['brand']

with gr.Blocks(theme='default', title=brand_info['organizationName']) as demo:
    gr.HTML(f'''<div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="{brand_info['logo']['title']}" alt="{brand_info['organizationName']} Logo" style="height: 100px;">
        </div>''')
    gr.ChatInterface(
        fn=ai_chatbot,
        chatbot=gr.Chatbot(height=500, avatar_images=(None, brand_info['chatbot']['avatar']), type="messages"),
        title=brand_info['organizationName'],
        description=f"{brand_info['slogan']} (RAG without Embeddings)",
        examples=[
            ["What is HERE AND NOW AI?"],
            ["What is the mission of HERE AND NOW AI?"],
            ["What courses does HERE AND NOW AI offer?"],
            ["Who is the CTO?"],
        ]
    )

if __name__ == "__main__":
    demo.launch()