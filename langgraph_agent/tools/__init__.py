# tools/__init__.py
"""
Tools registry for the Research Copilot.

Exports:
- TOOLS: list[BaseTool] ready to pass into your agent/router
- TOOL_REGISTRY: dict[name -> tool]
- get_tool(name): convenience accessor
- list_tool_names(): returns available tool names

Notes:
- Tavily MCP expects TAVILY_MCP_URL in your env (or TAVILY_API_KEY if spawning locally).
- Wikipedia tools require no API key.
- RAG uses a persistent Chroma collection and local Ollama embeddings.
"""

from __future__ import annotations
from typing import Dict, List
from langchain_core.tools import BaseTool

# ---- Web search (MCP-backed Tavily) ----
from .web_search_mcp import (
    web_search_tool,
    WebSearchInput,
    WebSearchOutput,
)

# ---- Web fetch (HTML/PDF → clean text) ----
from .web_fetch import (
    web_fetch_tool,
    WebFetchInput,
    WebFetchOutput,
)

# ---- NTRS (NASA Technical Reports Server) ----
from .ntrs_search import (
    ntrs_search_tool,
    NTRSSearchInput,
    NTRSSearchOutput,
)

# ---- Wikipedia (lookup + extract) ----
from .wikipedia import (
    wikipedia_lookup_tool,
    wikipedia_extract_tool,
    WikiLookupInput,
    WikiLookupOutput,
    WikiExtractInput,
    WikiExtractOutput,
)

# ---- RAG retrieval (Chroma + Ollama embeddings) ----
from .rag_retrieve import (
    rag_retrieve_tool,
    RagRetrieveInput,
    RagRetrieveOutput,
)

# ---- Public registry ----
TOOLS: List[BaseTool] = [
    ntrs_search_tool,
    web_search_tool,
    web_fetch_tool,          # placed after search so the agent can fetch results it finds
    wikipedia_lookup_tool,
    wikipedia_extract_tool,
    rag_retrieve_tool,
]

TOOL_REGISTRY: Dict[str, BaseTool] = {t.name: t for t in TOOLS}

def get_tool(name: str) -> BaseTool:
    """Return a tool by its registered name, e.g., 'web_search'."""
    return TOOL_REGISTRY[name]

def list_tool_names() -> List[str]:
    """List available tool names."""
    return list(TOOL_REGISTRY.keys())

__all__ = [
    # tools
    "ntrs_search_tool",
    "web_search_tool",
    "web_fetch_tool",
    "wikipedia_lookup_tool",
    "wikipedia_extract_tool",
    "rag_retrieve_tool",
    # schemas
    "NTRSSearchInput",
    "NTRSSearchOutput",
    "WebSearchInput",
    "WebSearchOutput",
    "WebFetchInput",
    "WebFetchOutput",
    "WikiLookupInput",
    "WikiLookupOutput",
    "WikiExtractInput",
    "WikiExtractOutput",
    "RagRetrieveInput",
    "RagRetrieveOutput",
    # registries/helpers
    "TOOLS",
    "TOOL_REGISTRY",
    "get_tool",
    "list_tool_names",
]
