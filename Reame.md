# 🌟 Disha — NHS Navigator

An AI-powered RAG chatbot that helps South Asian migrants and international students navigate the UK's National Health Service (NHS) — built as part of **Project Disha**, a broader platform supporting the South Asian diaspora through arrival, settlement, and integration in Western countries.

**🔗 Live App:** [disha-nhs-navigator.streamlit.app](https://disha-nhs-navigator-uwvagj7fzgymvymp93jadk.streamlit.app/)

---

## 💡 Why This Project

Navigating a new country's healthcare system is one of the most confusing and high-stakes parts of relocating abroad. NHS terminology, registration steps, and emergency protocols are often unclear to newcomers — especially international students and migrants from South Asia.

Disha answers NHS-related questions in plain, warm language, grounded entirely in **official NHS documentation** — so answers are accurate, not hallucinated.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` |
| Framework | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Frontend | Streamlit |
| Data Source | Official NHS.uk pages (12 sources) |

---

## ⚙️ How It Works

1. **Ingestion** — Official NHS web pages are loaded and split into overlapping chunks
2. **Embedding** — Each chunk is converted into a vector using HuggingFace embeddings
3. **Retrieval** — On a user query, the 4 most relevant chunks are retrieved via FAISS similarity search
4. **Generation** — Groq's LLaMA 3.1 model generates a warm, accurate answer grounded in retrieved context
5. **Citation** — Every response displays its source NHS page(s) for transparency and trust

This is a **Retrieval-Augmented Generation (RAG)** architecture — the model only answers from real NHS content, reducing hallucination risk.

---

## 📋 What You Can Ask Disha

- GP registration & what to expect from your surgery
- NHS 111 & when to go to A&E
- Mental health support services
- Student healthcare access
- Prescriptions & pharmacies
- Finding an NHS dentist
- Maternity & antenatal care
- The NHS App

---

## 🚀 Running Locally

```bash
git clone https://github.com/sumitg22/disha-nhs-navigator.git
cd disha-nhs-navigator
pip install -r requirements.txt
```

Add your Groq API key to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Run the app:
```bash
streamlit run app.py
```

---

## 🌏 About Project Disha

Disha (Sanskrit for *direction/path*) is a broader initiative aimed at building AI-powered tools that help the South Asian diaspora navigate life in the UK and other Western countries — from healthcare and housing to employment and financial systems. This NHS Navigator is the first working module of that vision.

---


**Sumit** — Computer Engineering graduate, incoming MSc Business Analytics student at Newcastle University. Background in consulting (EY) and social media analytics. Building toward a career bridging data, healthcare, and India–UK markets.

[GitHub](https://github.com/sumitg22)
