# learn-to-build-ai-chatbots-with-langchain-poml-gradio-tutorials-for-beginners

![HERE AND NOW AI Logo](https://raw.githubusercontent.com/hereandnowai/images/refs/heads/main/logos/logo-of-here-and-now-ai.png)

**Designed with passion for innovation** - HERE AND NOW AI

Tutorial series for building AI chatbots using LangChain, POML, and Gradio frameworks

## Core Technologies
These projects are built on a modern stack, prioritizing ease of use, power, and scalability.

Core Logic: LangChain

AI Models: Google Gemini (via the langchain-google-genai library)

User Interface: Gradio

Local Embeddings: Hugging Face Sentence Transformers

Vector Storage: FAISS by Meta AI

## Project Showcase
This repository is structured as a series of standalone projects, each in its own directory.

### 1. AI Chatbot without Memory
Description: A foundational chatbot that answers questions one at a time. It does not remember previous turns in the conversation.

Key Learning: Basic integration of a language model with a Gradio UI.

### 2. AI Chatbot with Memory
Description: An enhanced chatbot that can remember the context of the current conversation, allowing for follow-up questions.

Key Learning: Implementing conversational memory using LangChain's RunnableWithMessageHistory.

### 3. RAGbot with a Text File
Description: A chatbot that answers questions based on the content of a local text file. This project uses the "context stuffing" method.

Key Learning: Basic Retrieval-Augmented Generation (RAG) by providing a knowledge source to the model.

### 4. RAGbot with a Live Web Page
Description: A RAG chatbot that scrapes a live website and uses its content as the knowledge base for answering questions.

Key Learning: Using LangChain's WebBaseLoader to fetch and process web content in real-time.

### 5. RAGbot with a PDF Document
Description: A chatbot that can read a PDF document and answer questions based on its contents, featuring a custom Gradio UI.

Key Learning: Handling PDF files with PyPDFLoader and building a more complex user interface.

### 6. RAGbot with a Local Vector DB
Description: An advanced and efficient RAG chatbot that uses a local vector database (FAISS) for fast and scalable information retrieval.

Key Learning: Creating embeddings with a local Hugging Face model to avoid API rate limits and building a persistent, searchable knowledge base.

### 7. Multimodal Image Describer
Description: A multimodal application that takes an uploaded image and generates a detailed text description.

Key Learning: Working with multimodal models like Gemini 1.5 Flash to process both text and image inputs.

## Getting Started
Follow these steps to set up and run any of the projects on your local machine.

### 1. Prerequisites
Python 3.9 or higher

Access to a Google AI Studio API key

### 2. Clone the Repository
```
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Set Up Your API Key
Create a file named .env in the root directory of the project. Add your Google API key to this file:
```
GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"
```

### 4. Install Dependencies
All required Python libraries for all seven projects are listed in the consolidated requirements.txt file. Install them using pip:
```
pip install -r requirements.txt
```
Note: The first time you run Project 7, it will download a local sentence transformer model (a few hundred MB), which may take a few minutes.

### 5. Run a Project
Navigate into the directory of the project you wish to run and execute its app.py file.

Example for Project 5 (PDF RAGbot):
```
cd project_5_ai_ragbot_with_pdf/
python app.py
```
This will launch a local Gradio server. Open the URL provided in your terminal (e.g., http://127.0.0.1:7860) in your web browser to interact with the chatbot.

This collection serves as a practical guide to building modern AI applications. We encourage you to explore, modify, and build upon these examples. Happy coding!

## License

MIT

## Tutorial Series

Find the full tutorial series at: [HERE AND NOW AI GitHub](https://github.com/hereandnowai/)

For more information, visit [HERE AND NOW AI](https://hereandnowai.com) or contact info@hereandnowai.com.
