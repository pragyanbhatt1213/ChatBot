from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult, ChatGeneration, ChatResult
from pydantic import Field
from together import Together
import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key and set it for Together client
TOGETHER_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_AI_API_KEY not found in environment variables")

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
Together.api_key = TOGETHER_API_KEY

# Sample text for initial vector store
INITIAL_TEXTS = [
    # Agra Museums and Monuments
    "The Taj Museum, located within the Taj Mahal complex in Agra, houses original architectural drawings of the Taj Mahal, ancient manuscripts, government decrees, and artifacts from the Mughal period.",
    "The Archaeological Museum in Agra Fort displays a rich collection of Mughal artifacts, including weapons, pottery, paintings, and architectural pieces from the 16th to 19th centuries.",
    "The Taj Mahal is a UNESCO World Heritage Site built by Emperor Shah Jahan in memory of his beloved wife Mumtaz Mahal. This white marble monument represents the pinnacle of Mughal architecture.",
    "Agra Fort, also known as the Red Fort of Agra, is a historical fortress and palace complex. It houses several museums and structures including Jahangir Palace, Khas Mahal, and Diwan-i-Khas.",
    
    # Bhopal Museums and Monuments
    "The State Museum Bhopal, established in 1965, houses extensive collections of sculptures, coins, paintings, and artifacts from different periods of Indian history.",
    "The Tribal Museum in Bhopal showcases the rich cultural heritage of Madhya Pradesh's tribal communities, featuring traditional art, crafts, and lifestyle exhibits.",
    "The Regional Science Centre in Bhopal combines educational exhibits with historical artifacts, featuring galleries on natural history and indigenous science.",
    "Bharat Bhavan in Bhopal is a multi-arts complex and museum that celebrates Indian tribal and contemporary art, featuring exhibitions of paintings, sculptures, and installations.",
    
    # Delhi Museums
    "The National Museum in Delhi houses over 200,000 artifacts spanning 5,000 years of Indian cultural heritage, including ancient sculptures, jewelry, paintings, and manuscripts.",
    "The National Gallery of Modern Art in Delhi showcases Indian modern art from the 1850s onwards, with works by prominent Indian artists.",
    
    # Kolkata Museums
    "The Indian Museum in Kolkata, founded in 1814, is the oldest and largest museum in India, featuring rare collections of antiques, armor, ornaments, fossils, and Mughal paintings.",
    "The Victoria Memorial in Kolkata houses a museum that depicts the history of British India through paintings, sculptures, and artifacts.",
    
    # Historical Context
    "Indian museums preserve artifacts dating from the Indus Valley Civilization (3300-1300 BCE) through the Mughal Empire (1526-1857) to modern India.",
    "Many Indian museums are housed in historical buildings and palaces, combining architectural heritage with their collections.",
    "The Archaeological Survey of India (ASI) maintains numerous site museums at important archaeological excavations and monuments across India.",
    
    # Cultural Significance
    "Indian museums showcase the country's diverse cultural heritage, including Hindu, Buddhist, Jain, Islamic, and colonial period artifacts.",
    "Museum collections in India often feature religious sculptures, miniature paintings, textiles, weapons, coins, and decorative arts.",
    "Many Indian museums offer guided tours in multiple languages and provide educational programs about India's history and culture."
]

class TogetherChatModel(BaseChatModel):
    client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", **kwargs):
        super().__init__(**kwargs)
        try:
            self.client = Together()
            self.client.models.list()
        except Exception as e:
            raise ValueError(f"Failed to initialize Together client: {str(e)}")
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "together-ai-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
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
                max_tokens=512,
                temperature=0.7,
                **kwargs
            )

            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("No valid response received from Together AI")

            choice = response.choices[0]
            
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                response_text = choice.message.content
            elif isinstance(choice, dict):
                if 'message' in choice and 'content' in choice['message']:
                    response_text = choice['message']['content']
                elif 'text' in choice:
                    response_text = choice['text']
                else:
                    raise ValueError(f"Unexpected response format: {choice}")
            else:
                response_text = getattr(choice, 'text', str(choice))

            if not response_text:
                raise ValueError("Empty response text received")

            message = AIMessage(content=response_text)
            generation = ChatGeneration(
                message=message,
                text=response_text,
                generation_info={"finish_reason": "stop"}
            )

            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"Error in _generate: {str(e)}")
            message = AIMessage(content="I apologize, but I encountered an error processing your request. Please try again.")
            generation = ChatGeneration(
                message=message,
                text=message.content,
                generation_info={"finish_reason": "error"}
            )
            return ChatResult(generations=[generation])

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Clean up existing vector store if it exists
vector_store_path = "vector_db"
if os.path.exists(vector_store_path):
    try:
        shutil.rmtree(vector_store_path)
        print(f"Removed existing vector store at {vector_store_path}")
    except Exception as e:
        print(f"Error removing vector store: {e}")

# Initialize vector store with initial texts
vector_db = Chroma.from_texts(
    texts=INITIAL_TEXTS,
    embedding=embedding_function,
    persist_directory=vector_store_path,
)

# Create system prompt
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""

ANSWER_TEMPLATE = """You are an expert guide specializing in Indian museums, monuments, and cultural heritage. Your role is to provide detailed, informative answers about Indian museums, their collections, historical monuments, and cultural artifacts. Use the following pieces of context to answer the question.

If asked about a specific location, please include:
- The historical significance of the place
- Notable collections or artifacts
- Architectural features (for monuments)
- Cultural importance
- Any interesting facts or stories

If you don't have specific information about what was asked, you can mention related museums or monuments in the same city or region, but clearly state that you're providing related information.

Context: {context}
Question: {question}

Answer in a knowledgeable and engaging way, as a cultural guide would:"""

# Create prompts
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
ANSWER_PROMPT = PromptTemplate(
    template=ANSWER_TEMPLATE,
    input_variables=["context", "question"]
)

# Update the chain configuration
def create_qa_chain():
    try:
        llm = TogetherChatModel()
        
        # Configure memory with explicit output_key
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Explicitly set which key to store
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": ANSWER_PROMPT
            },
            chain_type="stuff",
            verbose=True
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        raise

# Initialize the QA chain
qa_chain = create_qa_chain()

def rag(question: str) -> str:
    try:
        if not question or not question.strip():
            return "Please provide a valid question."
            
        # Use invoke instead of deprecated __call__
        response = qa_chain.invoke({"question": question})
        
        # Extract just the answer from the response
        answer = response.get("answer", "I apologize, but I couldn't generate an answer at this moment.")
        return answer
    except Exception as e:
        print(f"Error in rag function: {str(e)}")
        return "I apologize, but I encountered an error processing your question. Please try again."