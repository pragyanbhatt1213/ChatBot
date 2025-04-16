from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from pydantic import Field
from together import Together
import os
from dotenv import load_dotenv

load_dotenv()

TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

class TogetherChatModel(BaseChatModel):
    client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

    def __init__(self, api_key: str = TOGETHER_AI_API_KEY, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", **kwargs):
        super().__init__(**kwargs)
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        together_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else \
                   "assistant" if isinstance(message, AIMessage) else \
                   "system" if isinstance(message, SystemMessage) else "user"
            together_messages.append({"role": role, "content": message.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=together_messages,
            stop=stop,
            **kwargs
        )

        message = AIMessage(content=response.choices[0].message.content)
        generation = ChatGeneration(message=message)
        return LLMResult(generations=[generation])

    def _llm_type(self) -> str:
        return "together-ai-chat"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="rich_dad_poor_dad",
    embedding_function=embedding_function,
)

# create prompt
QA_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the user question.
chat_history: {chat_history}
Context: {text}
Question: {question}
Answer:""",
    input_variables=["text", "question", "chat_history"]
)

# create chat model
llm = TogetherChatModel()

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history")

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
    chain_type="refine",
)

def get_conversation_chain():
    return qa_chain

# question
question = "What is the book about?"

# call QA chain
response = qa_chain({"question": question})

print(response.get("answer"))