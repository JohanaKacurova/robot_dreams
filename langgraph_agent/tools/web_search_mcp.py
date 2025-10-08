# tools/web_search_mcp.py
from __future__ import annotations

import os, time, json, asyncio
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()  # make .env vars available even in `python -c`

import requests
from pydantic import BaseModel, Field, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import StructuredTool

# ---- tldextract WITHOUT network (avoid PSL download delays) ----
import tldextract
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)


# =========================
# Schemas
# =========================
class SearchResult(BaseModel):
    url: HttpUrl
    title: str
    snippet: str = ""
    score: float = 0.0

class WebSearchInput(BaseModel):
    q: str = Field(..., description="Natural language query.")
    k: int = Field(5, ge=1, le=10)
    site_allow: Optional[List[str]] = None
    site_block: Optional[List[str]] = None
    recency_days: Optional[int] = Field(None, ge=1)

class WebSearchOutput(BaseModel):
    results: List[SearchResult]
    latency_ms: int


# =========================
# Helpers
# =========================
def _domain_key(url: str) -> str:
    ext = _EXTRACTOR(url)
    return ".".join(p for p in [ext.domain, ext.suffix] if p)

def _passes(url: str, allow: Optional[List[str]], block: Optional[List[str]]) -> bool:
    host = _domain_key(url).lower()
    if allow and not any(host.endswith(s.lstrip(".").lower()) or host == s.lower() for s in allow):
        return False
    if block and any(host.endswith(s.lstrip(".").lower()) or host == s.lower() for s in block):
        return False
    return True

def _pick_search_tool_name(tools) -> str:
    tool_list = getattr(tools, "tools", tools)
    names = []
    for t in tool_list:
        n = getattr(t, "name", None)
        if n is None and isinstance(t, dict):
            n = t.get("name")
        names.append(n)
    for cand in ("search", "tavily-search", "web_search"):
        if cand in names:
            return cand
    for n in names:
        if n and "search" in n.lower():
            return n
    raise RuntimeError(f"No search-like tool exposed by MCP server. Available: {names}")

def _parse_tool_result(resp) -> List[dict]:
    """Normalize MCP ToolResponse into list[dict] items with {url,title,content,score?}."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict) and "results" in resp:
        return resp["results"]

    content = getattr(resp, "content", None)
    out: List[dict] = []
    if content:
        for c in content:
            txt = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
            if not txt:
                continue
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
                    out.extend(obj["results"])
                elif isinstance(obj, list):
                    out.extend(obj)
                else:
                    out.append({"url": "", "title": "", "content": txt, "score": 0})
            except Exception:
                out.append({"url": "", "title": "", "content": txt, "score": 0})
    return out


# =========================
# REST backend (fast & robust)
# =========================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def _normalize_rest(obj) -> List[dict]:
    # Tavily may return {"results":[...]} or {"sources":[...]} depending on endpoint/flags
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
        if "sources" in obj and isinstance(obj["sources"], list):
            out = []
            for s in obj["sources"]:
                out.append({
                    "url": s.get("url") or s.get("source") or "",
                    "title": s.get("title") or s.get("url") or "",
                    "content": s.get("snippet") or s.get("content") or "",
                    "score": s.get("score") or 0.0,
                })
            return out
    if isinstance(obj, list):
        return obj
    return []

@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=0.5, min=0.5, max=1.0))
def _tavily_rest_search(args: WebSearchInput) -> List[dict]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set. Put it in .env if using REST backend.")
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": args.q,
        "max_results": max(1, min(args.k * 3, 20)),
        "search_depth": "basic",
        "include_answer": False,
        "include_domains": args.site_allow or None,
        "exclude_domains": args.site_block or None,
    }
    if args.recency_days:
        payload["days"] = args.recency_days

    # Try modern endpoint first; fall back if needed
    urls = [
        "https://api.tavily.com/search",
        "https://api.tavily.com/query",
    ]
    last_exc = None
    for u in urls:
        try:
            r = requests.post(u, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            return _normalize_rest(r.json())
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Tavily REST failed: {last_exc}")


# =========================
# Remote MCP backend (SSE)
# =========================
@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=0.6, min=0.6, max=1.2))
def _tavily_mcp_search(args: WebSearchInput) -> List[dict]:
    url = os.getenv("TAVILY_MCP_URL")
    if not url:
        raise RuntimeError("TAVILY_MCP_URL is not set (needed for MCP backend).")

    # Lazy import so this file can import even if MCP SDK missing
    from mcp.client.sse import sse_client
    from mcp import ClientSession

    async def _run():
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                # initialize is optional
                try:
                    await asyncio.wait_for(session.initialize(), timeout=4)
                except Exception:
                    pass

                tools = await asyncio.wait_for(session.list_tools(), timeout=6)
                tool_name = _pick_search_tool_name(tools)

                payload = {
                    "query": args.q,
                    "max_results": max(1, min(args.k * 3, 20)),
                }
                if args.recency_days:
                    payload["days"] = args.recency_days
                if args.site_allow:
                    payload["include_domains"] = args.site_allow
                if args.site_block:
                    payload["exclude_domains"] = args.site_block

                resp = await asyncio.wait_for(session.call_tool(tool_name, payload), timeout=10)
                return _parse_tool_result(resp)

    try:
        return asyncio.run(_run())
    except asyncio.TimeoutError:
        raise RuntimeError("Tavily MCP timed out (initialize/list_tools/call_tool). Check network/VPN/firewall.")


# =========================
# Backend chooser (REST first; MCP optional)
# =========================
def _do_search(args: WebSearchInput) -> List[dict]:
    backend = (os.getenv("WEB_SEARCH_BACKEND") or "").lower().strip()
    if backend == "rest":
        return _tavily_rest_search(args)
    if backend == "mcp":
        return _tavily_mcp_search(args)

    # auto mode: prefer REST if API key present; else MCP if URL present
    if TAVILY_API_KEY:
        try:
            return _tavily_rest_search(args)
        except Exception:
            # optional auto-fallback to MCP if configured
            if os.getenv("TAVILY_MCP_URL"):
                return _tavily_mcp_search(args)
            raise
    if os.getenv("TAVILY_MCP_URL"):
        return _tavily_mcp_search(args)
    raise RuntimeError("No backend configured. Set WEB_SEARCH_BACKEND=rest (and TAVILY_API_KEY) or WEB_SEARCH_BACKEND=mcp (and TAVILY_MCP_URL).")


# =========================
# Main tool impl (dedupe + top-k)
# =========================
def _web_search_impl(args: WebSearchInput) -> WebSearchOutput:
    t0 = time.time()
    raw = _do_search(args)

    seen = set()
    out: List[SearchResult] = []
    for item in raw:
        url = item.get("url") or ""
        if not url:
            continue
        if not _passes(url, args.site_allow, args.site_block):
            continue
        dom = _domain_key(url)
        if dom in seen:
            continue
        seen.add(dom)
        out.append(
            SearchResult(
                url=url,
                title=item.get("title") or url,
                snippet=(item.get("content") or "")[:400],
                score=float(item.get("score") or 0.0),
            )
        )
        if len(out) >= args.k:
            break

    return WebSearchOutput(results=out, latency_ms=int((time.time() - t0) * 1000))


# =========================
# kwargs wrapper for StructuredTool
# =========================
def _web_search(**kwargs) -> WebSearchOutput:
    args = WebSearchInput(**kwargs)
    return _web_search_impl(args)

web_search_tool = StructuredTool.from_function(
    name="web_search",
    description="Web search via Tavily (REST or MCP). Returns deduped results with title, url, snippet, score.",
    func=_web_search,
    args_schema=WebSearchInput,
    return_schema=WebSearchOutput,
)


# Optional: quick CLI smoke test
if __name__ == "__main__":
    import sys
    q = "site:nasa.gov star tracker" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    res = _web_search_impl(WebSearchInput(q=q, k=3))
    print(f"hits={len(res.results)} latency_ms={res.latency_ms}")
    for r in res.results:
        print("-", r.title, "->", r.url)
