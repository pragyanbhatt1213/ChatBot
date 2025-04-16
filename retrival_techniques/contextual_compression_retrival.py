from together import Together
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import EmbeddingsFilter
from decouple import config

# Define text and metadata
TEXT = [
    "Python is a versatile and widely used programming language.",
    "It is used in web development, AI, data science, and more.",
    "Python has a large community and many useful libraries.",
    "It is popular for automation and machine learning applications."
]
meta_data = [
    {"source": "document 1", "page": 1},
    {"source": "document 2", "page": 2},
    {"source": "document 3", "page": 3},
    {"source": "document 4", "page": 4}
]

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector database
vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

# Load API Key
TOGETHER_AI_API_KEY = config("TOGETHER_AI_API_KEY")

# Initialize Together AI client
client = Together(api_key=TOGETHER_AI_API_KEY)

# Function to use Together API directly
def process_query_with_together(query, documents):
    # First get relevant documents using vector search
    docs = vector_db.similarity_search(query, k=2)
    
    # Format the retrieved documents
    doc_texts = [f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)]
    doc_content = "\n".join(doc_texts)
    
    # Create a prompt for the Together API
    prompt = f"""
    Based on the following documents, answer the question: "{query}"
    
    {doc_content}
    
    Extract only the relevant information from these documents to answer the question.
    """
    
    # Call Together API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Print the response
    return response.choices[0].message.content

# Execute the query
query = "What areas is Python mostly used?"
answer = process_query_with_together(query, TEXT)

print("Query:", query)
print("\nAnswer:", answer)