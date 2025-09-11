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
from langchain_community.document_loaders import PyPDFLoader

# --- 1. Load Environment Variables & Branding ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("🔴 GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file.")

# Load branding data from branding.json
with open('branding.json', 'r') as f:
    brand = json.load(f)['brand']

# --- 2. Download and Load the PDF using LangChain ---
PDF_FILE_NAME = "About_HERE_AND_NOW_AI.pdf"

# Download the file if it doesn't exist locally
if not os.path.exists(PDF_FILE_NAME):
    print(f"Downloading PDF to {PDF_FILE_NAME}...")
    url = "https://raw.githubusercontent.com/hereandnowai/rag-workshop/main/pdfs/About_HERE_AND_NOW_AI.pdf"
    response = requests.get(url)
    with open(PDF_FILE_NAME, "wb") as f:
        f.write(response.content)

# Use LangChain's PyPDFLoader to load the document
try:
    loader = PyPDFLoader(PDF_FILE_NAME)
    docs = loader.load()
    # Extract the text content from the loaded pages
    pdf_context = "\n\n".join(doc.page_content for doc in docs)
    print("✅ Successfully loaded PDF context.")
except Exception as e:
    print(f"🔴 Error reading PDF: {e}")
    pdf_context = "Error: Could not read the PDF context."


# --- 3. Create the LangChain RAGbot ---
# Use the model from your project_5/app.py file
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)

# The prompt template includes the entire PDF context
qa_system_prompt = f"""You are Caramel AI, an AI assistant for HERE AND NOW AI.
Answer the user's questions based only on the following context retrieved from the PDF document.
Keep your answers concise and accurate.

Context:
---
{pdf_context}
---
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Create the chain with memory
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

# --- 5. Gradio Interface Function ---
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

# --- 6. Build the Custom Gradio UI ---
if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="yellow", secondary_hue="teal")) as demo:
        with gr.Row():
            # Sidebar Column for Branding
            with gr.Column(scale=1, elem_id="sidebar"):
                gr.Image(brand['logo']['favicon'], height=150, width=150)
                gr.Markdown(f"## {brand['organizationName']}")
                gr.Markdown(f"*{brand['slogan']}*")
                gr.Markdown("---")
                gr.Markdown("### Connect with us")
                for name, link in brand['socialMedia'].items():
                    gr.Markdown(f"[{name.capitalize()}]({link})")

            # Main Column for the Chatbot
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=ai_chatbot,
                    chatbot=gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        avatar_images=(None, brand['chatbot']['avatar']),
                        height=600
                    ),
                    title="AI Assistant",
                    description="Ask me anything about HERE AND NOW AI based on the provided PDF document.",
                    examples=[
                        ["Where is HERE AND NOW AI located?"],
                        ["What is the mission of HERE AND NOW AI?"],
                        ["Who is the CTO?"],
                        ["What services does HERE AND NOW AI provide?"],
                        ["What kind of courses are offered?"]
                    ]
                )
    demo.launch()