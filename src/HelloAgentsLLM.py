import os
import sys
import json
import importlib
import traceback
import litellm
from litellm import completion
from litellm.integrations.custom_logger import CustomLogger
from dotenv import load_dotenv

from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()
import logging

litellm.set_verbose = False
litellm.suppress_debug_info = True

# --- JSON request/response file logger ---
_LOG_FILE = os.path.join(os.path.dirname(__file__), "litellm_io.log")

class _JsonFileLogger(CustomLogger):
    """Logs the full JSON request and response for every LiteLLM call."""

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, error=True)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj, start_time, end_time, error=True)

    def _write(self, kwargs, response_obj, start_time, end_time, error=False):
        try:
            entry = {
                "timestamp": start_time.isoformat(),
                "duration_s": round((end_time - start_time).total_seconds(), 3),
                "error": error,
                "request": {
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages"),
                    "optional_params": {
                        k: v for k, v in kwargs.get("optional_params", {}).items()
                        if k != "stream"
                    },
                },
                "response": (
                    response_obj.model_dump() if hasattr(response_obj, "model_dump")
                    else str(response_obj)
                ),
            }
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

litellm.callbacks = [_JsonFileLogger()]

# To enable UTF8 unicode to be redirectable via tee in windows terminal
try:
    if hasattr(sys, "stdout") and sys.stdout is not None:
        enc = (sys.stdout.encoding or "").lower()
        if "utf" not in enc:
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                os.environ.setdefault("PYTHONUTF8", "1")
                os.environ.setdefault("PYTHONIOENCODING", "utf-8")
except Exception:
    pass

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

