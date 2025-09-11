import gradio as gr
import os
from dotenv import load_dotenv
import base64

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# --- 1. Load Environment Variables & Initialize Model ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("🔴 GOOGLE_API_KEY or GEMINI_API_KEY not found in .env file.")

# Initialize the multimodal Gemini model via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

# --- 2. Helper function to encode the image ---
def image_to_base64(image_path):
    """Converts an image file to a Base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# --- 3. The main function to be called by Gradio ---
def get_image_description(image_path):
    """
    Takes an image path, creates a multimodal message,
    and gets the description using a LangChain chain.
    """
    if image_path is None:
        return "Please upload an image first."

    # Encode the uploaded image to Base64
    base64_image = image_to_base64(image_path)

    # Create the multimodal message using LangChain's HumanMessage
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe this image in detail. Be observant and thorough.",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
        ]
    )
    
    # Create a simple chain and invoke it
    chain = llm | StrOutputParser()
    response = chain.invoke([message])
    
    return response

# --- 4. Create and Launch the Gradio UI ---
if __name__ == "__main__":
    gr.Interface(
        fn=get_image_description,
        inputs=gr.Image(type="filepath", label="Upload Image"),
        outputs=gr.Textbox(label="Image Description", lines=10),
        title="AI Image Describer",
        description="Upload an image and the AI will describe it for you using LangChain and Gemini 1.5 Flash.",
        theme='monochrome'
    ).launch()