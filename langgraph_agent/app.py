# app.py
# ReAct Research Copilot with LangGraph + Ollama (Mistral)
# Tools: web_search (MCP Tavily), wikipedia_lookup/extract, rag_retrieve (Chroma)

from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any
import json, os

from pathlib import Path

def load_prompt(rel_path: str, default: str) -> str:
    p = Path(__file__).parent / "prompts" / rel_path
    return p.read_text(encoding="utf-8") if p.exists() else default

# default fallback (kept short)
DEFAULT_SYSTEM = "You are a ReAct Research Copilot. Use tools. Answer concisely with citations."

# if you want to inject dynamic tool names from your registry:
from tools import list_tool_names
raw = load_prompt("system_react.txt", DEFAULT_SYSTEM)
SYSTEM_REACT = raw.replace("{tool_names}", ", ".join(list_tool_names()))


from dotenv import load_dotenv
load_dotenv()  # loads .env (or load a specific file if you keep envs under ./.env/)

# ---- LLM ---------------------------------------------------------------------
try:
    # Preferred: keep your existing config in llm.py
    from llm import llm  # your ChatOllama(...) instance
except Exception:
    # Fallback (uncomment if you don't have llm.py):
    # from langchain_ollama import ChatOllama
    # llm = ChatOllama(
    #     model="mistral:instruct",
    #     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    #     temperature=0.2,
    #     num_ctx=4096,
    #     num_predict=512,
    #     top_p=0.9,
    #     top_k=40,
    #     repeat_penalty=1.1,
    # )
    raise

# ---- LangGraph ---------------------------------------------------------------
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

# ---- Tools registry ----------------------------------------------------------
# You already created these files.
from tools import (
    TOOLS,                       # list of BaseTool
    TOOL_REGISTRY,               # dict name -> tool
    # Schemas (handy for typed payloads if you build them in Python)
    WebSearchInput, WikiLookupInput, WikiExtractInput, RagRetrieveInput,
)

# ============== Agent state ==============
class State(TypedDict):
    messages: List  # LC messages
    steps: int
    max_steps: int
    pending_tool: Optional[Dict[str, Any]]  # {"tool": str, "input": dict}
    # optional extras you may want to keep:
    # evidence: List[dict]
    # citations: List[str]

# ============== System prompt ==============
SYSTEM_REACT = """You are a **ReAct Research Copilot**.
You can call one tool at a time from this set: web_search, wikipedia_lookup, wikipedia_extract, rag_retrieve.

TO CALL A TOOL: reply with ONLY this JSON (no prose):
{ "tool": "<name>", "input": { ... } }

Good defaults:
- web_search: { "q": "<query>", "k": 5, "recency_days": 365 }
- wikipedia_lookup: { "q": "<query>", "k": 3, "lang": "en" }
- wikipedia_extract: { "title": "<exact page title>", "sections": ["Lead"], "lang": "en" }
- rag_retrieve: { "q": "<query>", "k": 5, "persist_dir": "chroma", "collection": "research_corpus" }

WHEN YOU ARE READY TO ANSWER:
Reply with:
FINAL:
<concise answer ≤ 200 words including key facts>

CITATIONS:
- <title or host> — <url or source>
- ...
"""

# ============== Agent node ==============
def agent_node(state: State) -> State:
    """
    Ask the LLM what to do next. It either returns a JSON tool call,
    or a 'FINAL:' answer. We stash tool calls in state['pending_tool'].
    """
    msgs = [SystemMessage(content=SYSTEM_REACT)] + state["messages"]
    ai: AIMessage = llm.invoke(msgs)
    content = (ai.content or "").strip()

    # Try to interpret as tool call JSON
    pending: Optional[Dict[str, Any]] = None
    if content.startswith("{") and '"tool"' in content:
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "tool" in obj and "input" in obj:
                pending = {"tool": obj["tool"], "input": obj["input"]}
        except Exception:
            pending = None

    new_messages = state["messages"] + [ai]
    new_state: State = {
        **state,
        "messages": new_messages,
        "pending_tool": pending,
    }
    return new_state

# ============== Tool executor node ==============
def tool_executor_node(state: State) -> State:
    """
    Executes state['pending_tool'] using your StructuredTool registry,
    appends a ToolMessage with the JSON result, and clears the pending call.
    """
    call = state.get("pending_tool")
    assert call, "No pending tool call found"

    tool_name: str = call["tool"]
    tool_input: Dict[str, Any] = call.get("input", {}) or {}

    if tool_name not in TOOL_REGISTRY:
        # If LLM asked for an unknown tool, tell it and stop.
        tm = ToolMessage(content=json.dumps({"error": f"Unknown tool: {tool_name}"}), name="error")
        return {**state, "messages": state["messages"] + [tm], "pending_tool": None, "steps": state["steps"] + 1}

    tool = TOOL_REGISTRY[tool_name]

    # Execute tool (StructuredTool will validate inputs via its args_schema)
    try:
        result = tool.invoke(tool_input)  # returns a Pydantic model or dict
        # Convert to plain JSON-serializable
        if hasattr(result, "model_dump_json"):
            payload = result.model_dump_json()
        elif hasattr(result, "dict"):
            payload = json.dumps(result.dict())
        else:
            payload = json.dumps(result)
        tm = ToolMessage(content=payload, name=tool_name)
    except Exception as e:
        tm = ToolMessage(content=json.dumps({"error": str(e)}), name=tool_name)

    # Clear the pending call, bump steps
    return {
        **state,
        "messages": state["messages"] + [tm],
        "pending_tool": None,
        "steps": state["steps"] + 1,
    }

# ============== Routing logic ==============
def route_from_agent(state: State):
    """
    If LLM requested a tool, go to tool_executor; otherwise END (assume final answer).
    Also stop if we hit max_steps to prevent infinite loops.
    """
    if state.get("steps", 0) >= state.get("max_steps", 6):
        return END
    return "tool_executor" if state.get("pending_tool") else END

# ============== Build the graph ==============
def build_app():
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)

    graph.set_entry_point("agent")
    # Conditional edge: after 'agent' either run tool_executor or finish
    graph.add_conditional_edges("agent", route_from_agent, {"tool_executor": "tool_executor", END: END})
    # After executing a tool, go back to the agent to decide next step
    graph.add_edge("tool_executor", "agent")

    return graph.compile()

# ============== Run (demo) ==============
if __name__ == "__main__":
    app = build_app()

    # Demo question; adjust to your language/domain as needed.
    question = "Summarize key criticisms of RAG systems since 2024 and name 2 authoritative sources."

    init_state: State = {
        "messages": [HumanMessage(content=question)],
        "steps": 0,
        "max_steps": 6,  # guardrail
        "pending_tool": None,
    }

    out = app.invoke(init_state)
    # Print the last AI message (should be FINAL: ... + CITATIONS)
    msgs = out["messages"]
    final = [m for m in msgs if isinstance(m, AIMessage)][-1]
    print(final.content)
