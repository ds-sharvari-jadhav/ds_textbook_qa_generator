# 📚 AI Q&A Generator for Data Science Textbooks

This project is an AI-powered system that ingests specific chapters from data science textbook PDF, understands its content, and generates high-quality, interview-style questions and answers. It features a Retrieval-Augmented Generation (RAG) pipeline running entirely on a local machine.

## ✨ Features

- **Advanced PDF Parsing:** Uses the Nougat model to convert complex scientific PDFs into structured Markdown, accurately preserving text and LaTeX formulas.
- **Retrieval-Augmented Generation (RAG):** Implements an end-to-end RAG pipeline to ensure answers are grounded in the textbook's content.
- **Local & Private:** Runs entirely on a local machine using Ollama and a quantized Phi-3 model, ensuring privacy and zero API costs.
- **Two Generation Modes:**
  - **Single Q&A:** Ask a specific question and get a context-aware answer.
  - **Topic Study Sets:** Provide a topic (e.g., "ROC Curve") and get a set of interview questions to prepare with.
- **Interactive UI:** A simple and intuitive web interface built with Streamlit.

## 🚀 Demo

![Application Demo](./src/assets/app-demo.png)

## 🛠️ Technology Stack & Architecture

- **Language:** Python
- **PDF Parsing:** Meta's Nougat (`facebook/nougat-small`)
- **LLM:** Microsoft's Phi-3 Mini (`phi3:mini-4k`) via Ollama
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Vector Store:** ChromaDB (local persistent storage)
- **UI:** Streamlit
- **Core Libraries:** PyMuPDF, Transformers, PyTorch, LangChain (for text splitting)

### Architecture Flow
1.  A chapter-specific PDF is processed by **Nougat** into a structured Markdown file.
2.  The Markdown is chunked, preserving metadata, and each chunk is converted into a vector embedding using **Sentence-Transformers**.
3.  These embeddings and chunks are stored in a **ChromaDB** vector store.
4.  When a user provides a query via the **Streamlit** UI, it is embedded and used to search **ChromaDB** for the most relevant context chunks (Retrieval).
5.  The retrieved context is formatted into a detailed prompt and sent to the local **Phi-3 LLM** via **Ollama**.
6.  The LLM generates a Q&A pair based *only* on the provided context (Generation).

## ⚙️ Setup & Installation

Follow these steps to set up and run the project locally.

**1. Prerequisites:**
- Python 3.10+
- [Ollama](https://ollama.ai/) installed on your machine.
- [Homebrew](https://brew.sh/) (on macOS for installing Tesseract).

**2. Clone the Repository:**
```bash
git clone https://github.com/ds-sharvari-jadhav/ds_textbook_qa_generator.git
cd ds_textbook_qa_generator

**3. Set Up Environment & Dependencies:**
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

**4.Set Up Local LLM:**
ollama pull phi3:mini-4k
