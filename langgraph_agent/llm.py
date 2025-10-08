# llm.py
# Central place to configure your LLM (Ollama Mistral).
# Reads sensible defaults from environment variables so you can tweak without editing code.

import os
from langchain_ollama import ChatOllama

MODEL = os.getenv("OLLAMA_MODEL", "mistral")           # e.g., "mistral" or "mistral:instruct"
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TEMPERATURE   = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
NUM_CTX       = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
NUM_PREDICT   = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
TOP_P         = float(os.getenv("OLLAMA_TOP_P", "0.9"))
TOP_K         = int(os.getenv("OLLAMA_TOP_K", "40"))
REPEAT_PENALTY= float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
SEED          = os.getenv("OLLAMA_SEED")  # optional

_extra = {}
if SEED is not None and SEED != "":
    try:
        _extra["seed"] = int(SEED)
    except ValueError:
        pass  # ignore bad seed values

# Export a single, reusable LLM instance
llm = ChatOllama(
    model=MODEL,
    base_url=BASE_URL,
    temperature=TEMPERATURE,
    num_ctx=NUM_CTX,
    num_predict=NUM_PREDICT,
    top_p=TOP_P,
    top_k=TOP_K,
    repeat_penalty=REPEAT_PENALTY,
    **_extra,
)
