# RAG-Mitra: India’s AI Heritage Guide

A smart conversational assistant built with **Retrieval-Augmented Generation (RAG)** to deliver rich, contextual insights into India's vast collection of museums, monuments, and cultural landmarks. Powered by cutting-edge language models and intelligent retrieval mechanisms.

---

## 🚀 Features

- 🏛️ **In-depth Knowledge**: Explore museums and monuments across India with detailed AI responses  
- 🤖 **Next-Gen AI**: Harnesses Together AI's powerful LLMs for accurate, human-like conversations  
- 💬 **Contextual Chat**: Maintains meaningful, flowing dialogue over multiple questions  
- 🔍 **Smart Search**: Retrieves only the most relevant pieces of information using semantic similarity  
- 📚 **Rich Cultural Content**: Learn about heritage sites with curated data  
- 🌆 **Pan-India Coverage**: Works for cities, states, and popular historical locations  

---

## 🛠️ Technology Stack

- **LangChain** – Framework for conversational logic  
- **Together AI** – Large Language Model backend  
- **HuggingFace** – Embeddings for vector search  
- **ChromaDB** – Vector database for fast similarity-based retrieval  
- **Streamlit** – Interactive web UI  
- **Python** – Core programming language  

---

## ⚙️ Prerequisites

- Python 3.8 or higher  
- Together AI API Key  
- Required packages from `requirements.txt`  

---

## 📦 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/pragyanbhatt1213/ChatBot.git
cd ChatBot
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Configure environment**  
Create a `.env` file in the root directory and add your API credentials:

```env
TOGETHER_API_KEY=your_key_here
```

---

## ▶️ Usage

Start the chatbot interface using Streamlit:  
```bash
streamlit run chatbot/main.py
```

---

## 🧱 System Architecture

### 1. **Vector Store (ChromaDB)**
- Stores embedded documents
- Enables semantic similarity search
- Auto-loads curated knowledge base

### 2. **Language Model (Together AI)**
- Powered by LLaMA 3.3 70B model  
- Customized for chat generation  
- Includes fallback/error management  

### 3. **Conversation Chain**
- Tracks message history  
- Context-aware response formulation  
- Retrieval-Augmented Generation (RAG) pipeline  

### 4. **Knowledge Base**
- Expertly curated info about Indian heritage  
- City-wise and theme-wise segmentation  
- Easily expandable for new data  

---

## 💡 Feature Highlights

### 🎯 Intelligent Response Generation
- Deep understanding of context  
- Historically and culturally accurate  
- Structurally rich responses  

### 🧠 Chat Memory
- Retains multi-turn conversations  
- Reformulates vague questions  
- Handles out-of-scope queries gracefully  

### 🔍 Semantic Retrieval
- Embedding-powered smart search  
- Prioritizes relevance  
- Filters noise from responses  

---

## 🤝 Contributing

Pull Requests are welcome! Whether it’s adding a new site or improving the code, your help is appreciated.

---

## 🙏 Acknowledgments

- **Together AI** – For their LLM API  
- **HuggingFace** – For open-source embeddings  
- **LangChain** – For orchestrating the conversation 
