import os
import sys
import importlib
import traceback
import litellm
from litellm import completion
from dotenv import load_dotenv

from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()
import logging

# Disable verbose output
litellm.set_verbose = False
# Suppress specific debug info
litellm.suppress_debug_info = True
# Optionally set logging level to WARN or ERROR
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

class HelloAgentsLLM:
    """
    A customized LLM client for the book "Hello Agents".
    It is used to call any service compatible with the OpenAI interface and uses streaming responses by default.
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        Initialize the client. Prioritize passed parameters; if not provided, load from environment variables.
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        print(f"🔧 Initializing HelloAgentsLLM with model: {self.model}")
        print(f"🔧 API Base URL: {baseUrl or 'Default (OpenAI)'}")
        
        if not self.model:
            raise ValueError("Model ID must be provided or defined in the .env file.")

        # If caller provided apiKey/baseUrl, expose them as common env vars so litellm picks them up.
        # This keeps behavior compatible with provider keys (OPENAI_API_KEY) and LiteLLM proxy usage.
        if apiKey:
            os.environ.setdefault("OPENAI_API_KEY", apiKey)
            os.environ.setdefault("OPENROUTER_API_KEY", apiKey)
            os.environ.setdefault("LITELLM_API_KEY", apiKey)
        if baseUrl:
            os.environ.setdefault("LITELLM_BASE_URL", baseUrl)
            os.environ.setdefault("OPENAI_API_BASE", baseUrl)

        # No persistent client required for the common `completion()` helper; keep timeout for later uses
        self._timeout = timeout

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
        """
        Call the large language model to think and return its response.
        Uses LiteLLM's Python SDK (`litellm.completion`) and supports streaming.
        """
        print(f"🧠 Calling {self.model} model via litellm...")
        try:
            response = completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # Handle streaming response (litellm returns OpenAI-style chunks)
            print("✅ Large language model response successful:")
            collected_content = []
            for chunk in response:
                # chunk may be a dict or an object with attributes depending on provider; handle both
                content = ""
                try:
                    # dict-like
                    content = (
                        (chunk.get("choices", [{}])[0].get("delta", {}) or {}).get("content")
                        or ""
                    )
                except Exception:
                    try:
                        # object-like
                        content = getattr(chunk.choices[0].delta, "content", "") or ""
                    except Exception:
                        content = ""

                if content:
                    print(content, end="", flush=True)
                    collected_content.append(content)
            print()  # Newline after streaming output ends
            return "".join(collected_content)

        except Exception as e:
            # litellm maps provider errors to OpenAI-like exceptions (AuthenticationError, APIError, etc.)
            print(f"❌ Error occurred when calling LLM API: {e}")
            return None

# --- Client Usage Example ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "Write a quicksort algorithm"}
        ]
        
        print("--- Calling LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- Complete Model Response ---")
            print(responseText)

    except ValueError as e:
        print(e)

