from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel
from decouple import config
from together import Together
from typing import Any, List, Dict, Optional

# Define text and metadata
TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

# Define prompt templates
combine_template = "Write a summary of the following text:\n\n{summaries}"
combine_prompt_template = PromptTemplate.from_template(
    template=combine_template)

question_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say \"thanks for asking!\" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
question_prompt_template = PromptTemplate.from_template(
    template=question_template)

# Define Together API ChatModel class
class TogetherChatModel(BaseChatModel):
    """Together AI chat model."""
    
    client: Any
    model_name: str
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """Initialize Together API client."""
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        super().__init__()
    
    def _generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate response using Together API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return {"generations": [{"message": response.choices[0].message.content}]}
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together-ai"

# Initialize Together LLM
llm = TogetherChatModel(api_key=config("TOGETHER_AI_API_KEY"))

# Create retriever chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'
    ),
    return_source_documents=True,
    chain_type="map_reduce",
    chain_type_kwargs={"question_prompt": question_prompt_template,
                       "combine_prompt": combine_prompt_template}
)

# Question
question = "What areas is Python mostly used"

# Call QA chain
response = qa_chain({"query": question})

print(response)

print("============================================")
print("====================Result==================")
print("============================================")
print(response["result"])

print("============================================")
print("===============Source Documents============")
print("============================================")
print(response["source_documents"][0])
