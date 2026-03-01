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

 Add you_groq_api_key in .env file : 

```bash
GROQ_API_KEY=your_api_key_here
```

---

### 4. Add PDF file

Place the pdf file in the root directory and rename it as "philippine_history":

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

## 🚀 Future Improvements

While the current implementation demonstrates a functional Retrieval-Augmented Generation (RAG) pipeline with a Streamlit-based UI, several enhancements can be made to evolve this into a production-grade AI system:

### 1. 🔍 Advanced Retrieval Optimization
- Implement **Hybrid Search (BM25 + Dense Embeddings)** to improve retrieval accuracy.
- Introduce **Cross-Encoder Re-ranking** to refine the relevance of retrieved chunks before passing them to the LLM.
- Dynamically tune `top_k` based on query complexity.

### 2. 📊 Evaluation & Metrics
- Integrate retrieval evaluation metrics such as **MRR (Mean Reciprocal Rank)** and **Recall@K**.
- Add **LLM response evaluation** for:
  - Faithfulness (grounded responses)
  - Hallucination detection
- Build automated evaluation pipelines for continuous performance monitoring.

### 3. ⚡ Scalability & Production Readiness
- Replace FAISS (in-memory) with scalable vector databases like **Pinecone, Weaviate, or ChromaDB**.
- Enable **distributed processing** for handling large-scale document ingestion.
- Containerize the application using Docker and deploy via cloud platforms (AWS/GCP/Azure).

### 4. 🧠 Embedding & Model Enhancements
- Experiment with **larger or domain-specific embedding models** for better semantic understanding.
- Introduce **multi-modal support** (text + tables/images from PDFs).
- Fine-tune models on domain-specific datasets for improved contextual relevance.

### 5. 🚀 UI/UX Enhancements (Streamlit)
- Add **chat history and session memory**.
- Support **multi-document upload and indexing**.
- Display **source citations and highlighted context chunks** used for answering queries.

### 6. 🔄 Caching & Performance Optimization
- Implement **embedding caching** to avoid recomputation.
- Use **response caching** for repeated queries.
- Optimize chunking strategies dynamically based on document structure.

### 7. 🤖 Agentic RAG Capabilities
- Extend the system to an **Agentic RAG architecture** using tools and decision-making workflows.
- Integrate external tools such as:
  - Web search APIs
  - Knowledge bases
- Enable multi-step reasoning and query decomposition.

### 8. 🔐 Security, Governance & Monitoring
- Add **input validation and sanitization** to prevent prompt injection.
- Implement **PII detection and masking** for sensitive data.
- Introduce **logging, monitoring, and alerting** mechanisms.
- Track model performance and enable **model drift detection** over time.

---

These enhancements aim to transform the current MVP into a **robust, scalable, and enterprise-ready AI system**, aligning with best practices in modern AI/ML engineering and deployment.

## 👨‍💻 Author

Abhishek Verma
