# tools/ntrs_search.py
from __future__ import annotations

import time
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import requests
from pydantic import BaseModel, Field, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import StructuredTool


# =========================
# Schemas
# =========================
class NTRSItem(BaseModel):
    title: str
    url: HttpUrl
    nasa_id: Optional[str] = None
    year: Optional[int] = None
    abstract: str = ""
    authors: List[str] = []

class NTRSSearchInput(BaseModel):
    q: str = Field(..., description="Query string for NASA Technical Reports Server.")
    k: int = Field(5, ge=1, le=20)
    year_from: Optional[int] = Field(None, description="Inclusive lower bound on publication year.")
    year_to: Optional[int] = Field(None, description="Inclusive upper bound on publication year.")
    sort: Optional[str] = Field("relevance", description="One of: 'relevance' or 'date'.")

class NTRSSearchOutput(BaseModel):
    results: List[NTRSItem]
    total: int
    latency_ms: int


# =========================
# Helpers
# =========================
def _coerce_year(v) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        s = str(v)
        # pick first 4 digits
        for i in range(len(s) - 3):
            chunk = s[i : i + 4]
            if chunk.isdigit():
                y = int(chunk)
                if 1800 <= y <= 2100:
                    return y
        return None
    except Exception:
        return None

def _normalize_items(data) -> tuple[List[NTRSItem], int]:
    """
    NTRS API responses differ by endpoint/version.
    This function tolerates multiple shapes and extracts a uniform list of NTRSItem.
    """
    items: List[NTRSItem] = []
    total = 0

    def to_item(src: dict) -> Optional[NTRSItem]:
        # unwrap _source if present
        if isinstance(src, dict) and "_source" in src and isinstance(src["_source"], dict):
            src = src["_source"]

        nasa_id = (
            src.get("nasa_id")
            or src.get("nasaId")
            or src.get("id")
            or src.get("nasaIdentifier")
            or None
        )
        title = src.get("title") or src.get("headline") or src.get("titleText") or "(untitled)"
        abstract = src.get("abstract") or src.get("summary") or src.get("description") or ""

        # year variants
        year = (
            src.get("publicationYear")
            or src.get("year")
            or src.get("pubYear")
            or src.get("publication_year")
        )
        year = _coerce_year(year)

        # authors as list[str]
        authors_field = src.get("authors") or src.get("author") or []
        authors: List[str] = []
        if isinstance(authors_field, str):
            authors = [a.strip() for a in authors_field.split(";") if a.strip()]
        elif isinstance(authors_field, list):
            acc: List[str] = []
            for a in authors_field:
                if isinstance(a, dict):
                    name = a.get("name") or a.get("authorName") or ""
                    if name:
                        acc.append(name)
                elif isinstance(a, str):
                    if a.strip():
                        acc.append(a.strip())
            authors = acc

        # canonical citation page
        url = f"https://ntrs.nasa.gov/citations/{nasa_id}" if nasa_id else "https://ntrs.nasa.gov/"
        try:
            return NTRSItem(
                title=title,
                url=url,  # HttpUrl field validates scheme/host
                nasa_id=nasa_id,
                year=year,
                abstract=(abstract or "")[:2000],
                authors=authors,
            )
        except Exception:
            return None

    # Accept several common response shapes
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            for it in data["items"]:
                obj = to_item(it)
                if obj:
                    items.append(obj)
            total = int(data.get("total", len(items)))
        elif "results" in data and isinstance(data["results"], list):
            for it in data["results"]:
                obj = to_item(it)
                if obj:
                    items.append(obj)
            total = int(data.get("total", len(items)))
        elif "hits" in data and isinstance(data["hits"], dict) and isinstance(data["hits"].get("hits"), list):
            for hit in data["hits"]["hits"]:
                obj = to_item(hit)
                if obj:
                    items.append(obj)
            total = int(data["hits"].get("total", {}).get("value", len(items)))
        else:
            # sometimes the top-level is already the item
            obj = to_item(data)
            if obj:
                items.append(obj)
            total = len(items)
    elif isinstance(data, list):
        for it in data:
            obj = to_item(it)
            if obj:
                items.append(obj)
        total = len(items)

    return items, total


# =========================
# Core request
# =========================
@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=0.5, min=0.5, max=1.0))
def _ntrs_http_search(args: NTRSSearchInput) -> dict:
    """
    Try known NTRS endpoints with tolerant parameters.
    No API key required.
    """
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ResearchCopilot/1.0 (+https://example.local)",
    }

    size = max(1, min(args.k * 3, 50))
    sort = "relevance" if (args.sort or "").lower() != "date" else "pub_date"

    # Endpoint candidates (try modern first)
    endpoints = [
        # Modern: POST with JSON body
        ("POST", "https://ntrs.nasa.gov/api/citations/search", {
            "q": args.q,
            "page": 1,
            "size": size,
            "sort": sort,
            "filters": {
                **({"yearStart": args.year_from} if args.year_from else {}),
                **({"yearEnd": args.year_to} if args.year_to else {}),
            },
        }),
        # Fallback: GET with query params
        ("GET", "https://ntrs.nasa.gov/api/citations", {
            "q": args.q,
            "page": 1,
            "size": size,
            "sort": sort,
            **({"yearStart": args.year_from} if args.year_from else {}),
            **({"yearEnd": args.year_to} if args.year_to else {}),
        }),
    ]

    last_err = None
    for method, url, payload in endpoints:
        try:
            if method == "POST":
                r = requests.post(url, headers=headers, json=payload, timeout=10)
            else:
                r = requests.get(url, headers=headers, params=payload, timeout=10)
            if r.status_code >= 400:
                last_err = RuntimeError(f"{method} {url} -> HTTP {r.status_code}")
                continue
            return r.json()
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"NTRS search failed. Last error: {last_err}")


# =========================
# Tool implementation
# =========================
def _ntrs_search_impl(args: NTRSSearchInput) -> NTRSSearchOutput:
    t0 = time.time()
    raw = _ntrs_http_search(args)
    items, total = _normalize_items(raw)

    # Sort if user requested date
    if (args.sort or "").lower() == "date":
        items.sort(key=lambda x: (x.year or -1), reverse=True)

    # Take top-k
    items = items[: args.k]

    return NTRSSearchOutput(
        results=items,
        total=total,
        latency_ms=int((time.time() - t0) * 1000),
    )


# =========================
# LangChain StructuredTool wrapper
# =========================
def _ntrs_search(**kwargs) -> NTRSSearchOutput:
    args = NTRSSearchInput(**kwargs)
    return _ntrs_search_impl(args)

ntrs_search_tool = StructuredTool.from_function(
    name="ntrs_search",
    description=(
        "Search NASA Technical Reports Server (NTRS) for domain-relevant papers/reports. "
        "Returns normalized items with title, URL, year, abstract, and authors. No API key required."
    ),
    func=_ntrs_search,
    args_schema=NTRSSearchInput,
    return_schema=NTRSSearchOutput,
)
