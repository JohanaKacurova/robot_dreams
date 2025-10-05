# main.py
# Entrypoint: loads env, builds the LangGraph app, runs a single query (or an interactive REPL).

import os
from dotenv import load_dotenv

# ---- Load environment ---------------------------------------------------------
# Default: load ./.env. If you keep env files under a folder like ./.env/dev.env, point to it:
# from pathlib import Path; load_dotenv(Path(__file__).parent / ".env" / "dev.env")
load_dotenv()

# ---- Build the app ------------------------------------------------------------
from app import build_app
from langchain_core.messages import HumanMessage

def run_once(question: str):
    app = build_app()
    state = {
        "messages": [HumanMessage(content=question)],
        "steps": 0,
        "max_steps": 6,
        "pending_tool": None,
    }
    out = app.invoke(state)
    # Find the last AI message and print it
    ai_msgs = [m for m in out["messages"] if m.type == "ai"]
    print(ai_msgs[-1].content if ai_msgs else "(no AI message)")

def repl():
    app = build_app()
    print("Research Copilot (ReAct). Type 'exit' to quit.")
    while True:
        try:
            q = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        state = {"messages": [HumanMessage(content=q)], "steps": 0, "max_steps": 6, "pending_tool": None}
        out = app.invoke(state)
        ai_msgs = [m for m in out["messages"] if m.type == "ai"]
        print("\n" + (ai_msgs[-1].content if ai_msgs else "(no AI message)"))

if __name__ == "__main__":
    # If you pass a question as an argument, answer once; otherwise start a small REPL.
    import sys
    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        repl()
