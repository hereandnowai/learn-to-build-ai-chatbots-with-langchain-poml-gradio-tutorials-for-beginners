import gradio as gr
import os
import json
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader

# --- 1. Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("🔴 GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file.")

# --- 2. Load Website Content using LangChain's WebBaseLoader ---
try:
    WEBSITE_URL = "https://hereandnowai.github.io/vac/"
    print(f"Fetching content from {WEBSITE_URL}...")
    loader = WebBaseLoader(WEBSITE_URL)
    docs = loader.load()
    # Extract the text content from the loaded documents
    website_context = "\n\n".join(doc.page_content for doc in docs)
    print("✅ Successfully loaded website context.")
except Exception as e:
    print(f"🔴 Error fetching website content: {e}")
    website_context = "Error: Could not load the website content."


# --- 3. Create the LangChain RAGbot ---
# Use the model from your original project_4/app.py file
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

# The prompt template includes the entire website context
qa_system_prompt = f"""You are Caramel AI, an AI assistant for HERE AND NOW AI.
Answer the user's questions based only on the following context retrieved from the website {WEBSITE_URL}.
Keep your answers concise and accurate.

Context:
---
{website_context}
---
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Create the chain
rag_chain = qa_prompt | llm

# --- 4. Add Message History for Conversational Memory ---
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
        description=f"{brand_info['slogan']} (RAG from Web)",
        examples=[
            ["What is HERE AND NOW AI?"],
            ["What courses are offered?"],
            ["Who is the CTO?"],
        ]
    )

if __name__ == "__main__":
    demo.launch()