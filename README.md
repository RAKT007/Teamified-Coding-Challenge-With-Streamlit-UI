# 💬 PDF RAG Chatbot (Streamlit Version)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot with a **Streamlit UI**.  
It allows users to ask questions from a PDF and get answers using semantic search + LLM.

---

## 🚀 Features

- Streamlit chat-based UI  
- PDF parsing using PyMuPDF  
- Text chunking using RecursiveCharacterTextSplitter  
- Transformer-based embeddings (HuggingFace)  
- Semantic search using FAISS  
- LLM-based answer generation (Groq)  
- Query intent detection (LLM-based)  
- Context-grounded answers (reduces hallucination)  
- FAISS index caching (faster performance)  
- Unit & integration testing using pytest  

---

## 📁 Project Structure

```
app.py          # Streamlit UI
tests/          # unit tests
requirements.txt
.env.example
```

---

## ⚙️ Setup Instructions

### 1. Create virtual environment

```bash
python -m venv myenv
```

Activate (Windows):

```bash
myenv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup environment variables

Create a `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```

(configs)

```bash
PDF_PATH=philippine_history.pdf
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
EMBED_MAX_LENGTH=256
INDEX_DIR=faiss_store
```

---

### 4. Add PDF file

Place the file in the root directory:

```
philippine_history.pdf
```

---

## ▶️ Run the Application

```bash
python -m streamlit run app.py

```

Then open the browser link (usually: http://localhost:8501)

---

## 💬 Usage

- Enter your question in the chat input  
- System will:
  1. Detect query intent  
  2. Retrieve relevant chunks from FAISS  
  3. Generate answer using Groq LLM  

- View retrieved chunks in **"Sources"** section  

---

## ♻️ Reset Index

- Use the **Reset Index** button in UI  
- This deletes FAISS cache  
- Index rebuilds automatically on next query  

---

## 🧪 Run Tests

Run all tests:

```bash
python -m pytest -q
```



---

## 🧠 Key Highlights

- End-to-end RAG pipeline  
- Intent-aware answer generation  
- Cached FAISS index for efficiency  
- Clean modular architecture  
- Fully tested system  

---

## 📌 Notes

- API keys are not hardcoded  
- Uses `.env` for configuration  
- Defaults provided for easy execution  
- Streamlit UI is an enhancement; core pipeline works independently  

---

## 👨‍💻 Author

Abhishek Verma
