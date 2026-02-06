# QA-Chatbot
Q&A Chatbot is a Retrieval-Augmented Generation (RAG) Q&A chatbot built with LangChain and powered by open-source Large Language Models (LLMs) via Ollama. It allows users to ask questions and receive accurate, context-aware answers based on previously embedded documents.
The system retrieves relevant information from documents using vector embeddings and injects that context into the LLM prompt to improve answer quality and reduce hallucinations.

### 🚀 Features
- 📄 Document-based (pdf)
- 🧠 RAG architecture (Retrieval + Generation)
- 🔍 Semantic search using vector embeddings
- 🤖 Powered by Ollama open-source LLMs
- 🔗 Built with LangChain
-  Local-first, privacy-friendly LLM inference
🏗️ Architecture Overview

### Documents are loaded and split into chunks
1. Chunks are embedded into vectors
2. Embeddings are stored in a vector database
3. User queries are embedded and matched with relevant chunks
4. Retrieved context is passed to the LLM for answer generation

### 🛠️ Tech Stack
- Python
- LangChain
- Ollama (open-source LLMs)
- Vector Store (Chroma )
- FastAPI / CLI / Script-based interface (depending on implementation)

### 📦 Installation
1. Clone the repository
```
git clone https://github.com/your-username/schoolai.git
cd schoolai
```
2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate # Linux / macOS
venv\Scripts\activate # Windows
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Install Ollama and pull some models
```
https://ollama.com

ollama pull gemma3:1b mxai-embed-large
```
