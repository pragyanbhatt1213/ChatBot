from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "../documents/dataset.pdf"

# Create loader
loader = PyPDFLoader(PDF_PATH)
# Split document
pages = loader.load_and_split()

# Define embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store (Chroma automatically persists)
vectordb = Chroma.from_documents(
    documents=pages,  # Fixed typo
    embedding=embedding_function,
    persist_directory="../vector_db",
    collection_name="dataset_done"
)

# No need to call vectordb.persist() manually!



# # embedding function
# embedding_func = SentenceTransformerEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )

# # create vector store
# vectordb = Chroma.from_documents(
#     documents=pages,
#     embedding=embedding_func,
#     persist_directory=f"../vector_db",
#     collection_name="rich_dad_poor_dad")

# # make vector store persistant
# vectordb.persist()