from crewai import BaseLLM
from typing import Any, Dict, List, Optional, Union
import requests

class CustomerLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, endpoint: str, temperature: Optional[float] = None):
        # IMPORTANT: Call super().__init__() with required parameters
        super().__init__(model=model, temperature=temperature)

        self.api_key = api_key
        self.endpoint = endpoint

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """Call the LLM with the given messages."""
        # Convert string to message format if needed
        if isinstance(messages, str):
            messages = [{"role": "user"}, "content": messages]
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        # Add tools if provided and supported
        if tools and self.supports_function_calling():
            payload["tools"] = tools

        # Make API call
        response = requests.post(
            self.endpoint, 
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )

        # Check for errors
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def supports_function_calling(self) -> bool:
        """Check if the model selected supports function calling"""
        # defualt return True
        return True
    
    def get_context_window_size(self) -> int:
        """Get the context window size for the model"""
        # defualt return 4096
        return 8192
    