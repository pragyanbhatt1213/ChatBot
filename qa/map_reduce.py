import os
import traceback
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from typing import Any, List, Optional
from together import Together, error as together_error
from pydantic import Field
from langchain_core.outputs import LLMResult, ChatGeneration

# --- Configuration ---
TOGETHER_AI_API_KEY="31bec9e04553c595531531922aad86fcf86cafc706f190d8033d37ef8e929aa6"

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# --- Data ---
TEXT = [
    "Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure.",
    "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
    "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility.",
    "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
    "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."
]

meta_data = [
    {"source": "document 1", "page": 1},
    {"source": "document 2", "page": 2},
    {"source": "document 3", "page": 3},
    {"source": "document 4", "page": 4},
    {"source": "document 5", "page": 5}
]

# --- Custom Langchain Chat Model for Together AI ---
class TogetherChatModel(BaseChatModel):
    client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default=DEFAULT_MODEL_NAME)

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL_NAME, **kwargs):
        super().__init__(**kwargs)
        if not api_key or api_key == "YOUR_TOGETHER_API_KEY_HERE":
            raise ValueError(
                "Valid Together AI API key is required. "
                "Set the TOGETHER_API_KEY environment variable or pass it directly."
            )
        try:
            self.client = Together(api_key=api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize Together client: {e}") from e
        self.model_name = model_name

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        together_messages = []
        for message in messages:
            role = "user"
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                print(f"Warning: Unknown message type {type(message)}, treating as user.")
            together_messages.append({"role": role, "content": message.content})

        try:
            print(f"Sending request to Together AI (Model: {self.model_name})...")
            api_kwargs = {"model": self.model_name, "messages": together_messages}
            if stop:
                api_kwargs["stop"] = stop
            api_kwargs.update(kwargs)

            response = self.client.chat.completions.create(**api_kwargs)

            if response.choices and response.choices[0].message:
                ai_text = response.choices[0].message.content
                print(f"Received response snippet: {ai_text[:70]}...")
                message_obj = AIMessage(content=ai_text)
                generation = ChatGeneration(message=message_obj)
            else:
                error_text = "Error: No valid response content received from API."
                print(error_text)
                message_obj = AIMessage(content=error_text)
                generation = ChatGeneration(message=message_obj)

            # Fix: Return a flat list of generations
            return LLMResult(generations=[generation])  # Changed from [[generation]] to [generation]

        except together_error.InvalidRequestError as e:
            print(f"Together API Invalid Request Error: {str(e)}")
            traceback.print_exc()
            raise e
        except Exception as e:
            print(f"Generic Error during Together API Call: {str(e)}")
            traceback.print_exc()
            raise e

    def _llm_type(self) -> str:
        return "together-ai-chat"

# --- Main Script Logic ---
def initialize_components():
    print("Initializing embedding function...")
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        raise

    print("Creating vector database...")
    try:
        vector_db = Chroma.from_texts(
            texts=TEXT,
            embedding=embedding_function,
            metadatas=meta_data
        )
    except Exception as e:
        print(f"Error creating Chroma vector DB: {e}")
        raise

    print("Initializing Together AI LLM...")
    if TOGETHER_AI_API_KEY == "YOUR_TOGETHER_API_KEY_HERE":
        print("--- WARNING ---")
        print("TOGETHER_API_KEY is not set or using a placeholder.")
        print("The script will likely fail when calling the API.")
        print(f"Attempting to use default model: {DEFAULT_MODEL_NAME}")
        print("Set the environment variable or replace the placeholder in the code.")
        print("---------------")

    try:
        llm = TogetherChatModel(api_key=TOGETHER_AI_API_KEY, model_name=DEFAULT_MODEL_NAME)
        print(f"Successfully initialized Together API client with model: {DEFAULT_MODEL_NAME}")
    except (ValueError, Exception) as e:
        print(f"Error initializing TogetherChatModel: {e}")
        raise

    return vector_db, llm

def run_simple_qa(llm, question, documents=TEXT):
    if not llm:
        print("LLM not initialized, skipping simple QA.")
        return "Error: LLM not available."

    print("\n--- Running Simple QA ---")
    prompt = f"""Please answer the following question based *only* on the text provided below. Do not use any external knowledge.

Provided Text:
{' '.join(documents)}

Question: {question}

Answer:"""

    try:
        print("Sending direct query to LLM...")
        messages = [HumanMessage(content=prompt)]
        response_message = llm.invoke(messages)

        # Debug to inspect response
        print(f"Debug: response_message type = {type(response_message)}, value = {response_message}")

        # Handle different response types
        if isinstance(response_message, LLMResult):
            if response_message.generations and response_message.generations[0]:
                generation = response_message.generations[0]
                if isinstance(generation, ChatGeneration):
                    answer = generation.message.content
                else:
                    answer = str(generation)
            else:
                answer = "Error: No valid generations in LLMResult."
        elif isinstance(response_message, AIMessage):
            answer = response_message.content
        else:
            print(f"Unexpected response type from LLM: {type(response_message)}")
            return "Error: Received unexpected response format from LLM."

        if answer.startswith("Error:"):
            print(f"LLM returned an error message: {answer}")
        return answer

    except Exception as e:
        print(f"Error during simple QA execution (caught from invoke): {e}")
        traceback.print_exc()
        return f"Error during simple QA execution: {str(e)}"

def run_retrieval_qa_chain(llm, vector_db, question):
    if not llm or not vector_db:
        print("LLM or Vector DB not initialized, skipping RetrievalQA chain.")
        return {"result": "Error: LLM or Vector DB not available.", "source_documents": []}

    print("\n--- Running RetrievalQA Chain ---")
    print("Creating retriever chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type="stuff",
        )
    except Exception as e:
        print(f"Error creating RetrievalQA chain: {e}")
        traceback.print_exc()
        return {"result": f"Error creating chain: {str(e)}", "source_documents": []}

    try:
        print("Executing query with chain...")
        response = qa_chain.invoke({"query": question})
        return response
    except Exception as error:
        print(f"Error executing query with chain (caught from invoke): {error}")
        return {"result": f"Error executing chain: {str(error)}", "source_documents": []}

# --- Main Execution ---
if __name__ == "__main__":
    vector_db, llm = None, None
    try:
        vector_db, llm = initialize_components()
    except Exception as e:
        print(f"Failed to initialize components: {e}. Exiting.")
        exit(1)

    question = "What areas is Python mostly used in, according to the provided text?"
    print(f"\nQuestion: {question}")

    # --- Run Simple QA ---
    simple_answer = run_simple_qa(llm, question)
    print("\n============================================")
    print("============== Simple Result ===============")
    print("============================================")
    print(simple_answer)
    print("============================================")

   