import gradio as gr
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv

# --- Setup and Configuration ---

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from environment variables
# Note: Your original file used "GEMINI_API_KEY", but "GOOGLE_API_KEY" is more standard
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


# Check if the API key is available
if not api_key:
    raise ValueError("🔴 GOOGLE_API_KEY or GEMINI_API_KEY not found. Please create a .env file and add your key.")

# --- Chatbot Configuration ---
try:
    # Using the model from your project_2/app.py file
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
except Exception as e:
    print(f"🔴 Error initializing the language model: {e}")
    llm = None

# Define the system prompt for the AI teacher
ai_teacher_prompt = """You are Caramel AI, an AI Teacher at HERE AND NOW AI - Artificial Intelligence Research Institute.
Your mission is to **teach AI to beginners** like you're explaining it to a **10-year-old**.
Always be **clear**, **simple**, and **direct**. Use **short sentences** and **avoid complex words**.
You are **conversational**. Always **ask questions** to involve the user.
After every explanation, ask a small follow-up question to keep the interaction going. Avoid long paragraphs.
Think of your answers as **one sentence at a time**. Use examples, analogies, and comparisons to things kids can understand.
Your tone is always: **friendly, encouraging, and curious**. Your answers should help students, researchers, or professionals who are just starting with AI.
Always encourage them by saying things like: "You’re doing great!" "Let’s learn together!" "That’s a smart question!"
Do **not** give long technical explanations. Instead, **build the understanding step by step.**
You say always that you are **“Caramel AI – AI Teacher, built at HERE AND NOW AI – Artificial Intelligence Research Institute.”**"""

# Create a chat prompt template that includes history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ai_teacher_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Create a chain that combines the prompt and the model
if llm:
    chain = prompt | llm

# --- Conversation History Management ---
# This dictionary will store the history for each user session
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Gets the conversation history for a given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create the runnable with message history, which gives the chatbot memory
if llm:
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

# --- Gradio Function ---

def ai_chatbot(message, history, session_id="main_session"):
    """
    This function is called by Gradio and uses the LangChain runnable to get a response.
    The `with_message_history` object automatically handles the conversation memory.
    """
    if not llm:
        return "🔴 Error: Language model not initialized. Check your API key and terminal for errors."

    try:
        # Invoke the LangChain runnable with memory
        response = with_message_history.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content
    except Exception as e:
        print(f"🔴 An error occurred: {e}")
        return "Sorry, something went wrong. Please check the terminal for the full error message."

# --- Gradio Interface ---

# Load branding data from the branding.json file
with open('branding.json') as f:
    brand_info = json.load(f)['brand']

# Create the Gradio interface using the code from your ui.py file
with gr.Blocks(theme='default', title=brand_info['organizationName']) as demo:
    gr.HTML(f'''<div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="{brand_info['logo']['title']}" alt="{brand_info['organizationName']} Logo" style="height: 100px;">
        </div>''')
    gr.ChatInterface(
        fn=ai_chatbot,
        chatbot=gr.Chatbot(height=500, avatar_images=(None, brand_info['chatbot']['avatar']), type="messages"),
        title=brand_info['organizationName'],
        description=brand_info['slogan'],
        examples=[
            ["What is AI?"],
            ["Can you explain machine learning?"],
            ["How does a neural network work?"],
        ]
    )

if __name__ == "__main__":
    demo.launch()