from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Any, Dict, List, Optional
from together import Together

class TogetherChatModel(BaseChatModel):
    """Together AI chat model."""

    client: Any = Field(default=None, exclude=True)
    model_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")

    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", **kwargs):
        """Initialize Together API client."""
        super().__init__(**kwargs)
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate response using Together API."""
        together_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                together_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                together_messages.append({"role": "assistant", "content": message.content})
            else:
                together_messages.append({"role": "system", "content": message.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=together_messages,
            **kwargs
        )

        message = AIMessage(content=response.choices[0].message.content)

        return {"generations": [{"message": message, "text": message.content}]}

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together-ai"
