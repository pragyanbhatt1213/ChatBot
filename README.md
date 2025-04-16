# Indian Museums and Monuments AI Guide

An intelligent chatbot system designed to provide detailed information about museums, monuments, and cultural artifacts across India. Built using state-of-the-art language models and retrieval-augmented generation.

## Features

- 🏛️ Comprehensive information about Indian museums and monuments
- 🤖 Advanced AI-powered responses using Together AI
- 💬 Natural conversation flow with context awareness
- 🔍 Intelligent retrieval of relevant information
- 📚 Rich knowledge base of cultural heritage
- 🌐 Support for multiple cities and historical sites

## Technology Stack

- **LangChain**: Framework for building the conversational AI system
- **Together AI**: Large Language Model provider
- **HuggingFace**: Embedding model for text processing
- **ChromaDB**: Vector database for efficient information retrieval
- **Streamlit**: User interface
- **Python**: Primary programming language

## Prerequisites

- Python 3.8 or higher
- Together AI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:



  
## Usage

Run the chatbot using Streamlit:
```bash
streamlit run chatbot/main.py
```

## System Architecture

### Components

1. **Vector Store**
- Uses ChromaDB for storing and retrieving document embeddings
- Automatically initializes with predefined knowledge base
- Efficient similarity search capabilities

2. **Language Model**
- Powered by Together AI's Llama-3.3-70B model
- Custom implementation for chat completion
- Robust error handling and response formatting

3. **Conversation Chain**
- Maintains chat history
- Contextual question answering
- Retrieval-augmented response generation

4. **Knowledge Base**
- Curated information about Indian museums and monuments
- Organized by cities and themes
- Easily extensible structure

## Features in Detail

### Intelligent Response Generation
- Context-aware answers
- Historical and cultural accuracy
- Structured information delivery

### Conversation Management
- Memory retention across chat sessions
- Question reformulation for better context
- Error handling and graceful fallbacks

### Information Retrieval
- Semantic search capabilities
- Relevant context extraction
- Dynamic response formation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Together AI for providing the language model
- HuggingFace for the embedding model
- LangChain for the framework

