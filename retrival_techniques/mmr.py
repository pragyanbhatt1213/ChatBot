from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]


emvedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb= Chroma.from_texts(
    texts=TEXT,
    embedding=emvedding_function,
    metadatas=meta_data
)

response = vectordb.max_marginal_relevance_search(
    query="What is Python?",k=2)

print(response)
# The above code creates a vector store using the Chroma library, which is used for semantic search.
# It uses the HuggingFaceEmbeddings model to convert text into embeddings and then performs a similarity search on the vector store.
# The response variable contains the top 2 most similar documents to the query "What is Python?" based on the embeddings.
# The response will contain the most relevant documents from the vector store that match the query.
# The output will be a list of documents that are semantically similar to the query.
# The documents will include the text and metadata associated with them.