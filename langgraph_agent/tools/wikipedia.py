# tools/wikipedia.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import StructuredTool

import wikipedia as wiki_search   # PyPI: wikipedia  (good search + summary)
import wikipediaapi               # PyPI: wikipedia-api (good section extraction)


# =========================
# Pydantic Schemas
# =========================
class WikiLookupInput(BaseModel):
    q: str = Field(..., description="Search query for Wikipedia.")
    k: int = Field(5, ge=1, le=10, description="Number of results to return.")
    lang: str = Field("en", description="Wikipedia language code (e.g., 'en', 'cs', 'sk').")

class WikiPage(BaseModel):
    page_id: Optional[str] = None
    title: str
    summary: str = ""
    url: Optional[HttpUrl] = None

class WikiLookupOutput(BaseModel):
    pages: List[WikiPage]

class WikiExtractInput(BaseModel):
    title: str = Field(..., description="Exact page title to extract from.")
    sections: Optional[List[str]] = Field(
        default=None,
        description="Section names to extract (case-insensitive). If omitted, returns lead summary.",
    )
    lang: str = Field("en", description="Wikipedia language code.")
    max_chars_per_section: int = Field(
        1200, ge=200, le=8000, description="Truncate each section to this many characters."
    )

class WikiSection(BaseModel):
    section: str
    text: str

class WikiExtractOutput(BaseModel):
    title: str
    page_id: Optional[str]
    url: Optional[HttpUrl]
    content: List[WikiSection]


# =========================
# Helpers
# =========================
def _set_lang(lang: str) -> str:
    """Set default language for the `wikipedia` package."""
    try:
        wiki_search.set_lang(lang)
    except Exception:
        wiki_search.set_lang("en")
        return "en"
    return lang

def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

def _flatten_sections(root: wikipediaapi.WikipediaPage, prefix="") -> List[wikipediaapi.WikipediaPage.Section]:
    """Depth-first list of sections with full path names."""
    out = []
    def walk(secs, path):
        for s in secs:
            name = s.title if not path else f"{path}/{s.title}"
            out.append((name, s))
            walk(s.sections, name)
    walk(root.sections, prefix)
    return out


# =========================
# Wikipedia LOOKUP
# =========================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(0.5, 0.5, 2.0))
def _lookup_impl(args: WikiLookupInput) -> WikiLookupOutput:
    lang = _set_lang(args.lang)

    # Search returns a list of titles; we over-fetch to counter disambiguations/missing pages.
    titles = wiki_search.search(query=args.q, results=min(args.k * 3, 20))
    pages: List[WikiPage] = []

    for t in titles:
        try:
            # Try to get a short summary and URL; skip disambiguation pages
            # sentences=2 keeps context small; adjust if you need more.
            summary = wiki_search.summary(t, sentences=2, auto_suggest=False, redirect=True)
            page = wiki_search.page(t, auto_suggest=False, redirect=True)
            # Some languages don't always return canonical URLs; guard with getattr.
            url = getattr(page, "url", None)
            pages.append(WikiPage(page_id=None, title=page.title, summary=summary, url=url))
        except wiki_search.exceptions.DisambiguationError:
            continue
        except wiki_search.exceptions.PageError:
            continue
        except Exception:
            continue

        if len(pages) >= args.k:
            break

    return WikiLookupOutput(pages=pages)


def _lookup_tool(**kwargs) -> WikiLookupOutput:
    args = WikiLookupInput(**kwargs)
    return _lookup_impl(args)



wikipedia_lookup_tool = StructuredTool.from_function(
    name="wikipedia_lookup",
    description=(
        "Search Wikipedia for relevant pages and return short summaries. "
        "Use to quickly ground definitions and background before deep web search."
    ),
    func=_lookup_tool,
    args_schema=WikiLookupInput,
    return_schema=WikiLookupOutput,
)


# =========================
# Wikipedia EXTRACT (sections)
# =========================
@retry(stop=stop_after_attempt(2), wait=wait_exponential(0.5, 0.5, 2.0))
def _extract_impl(args: WikiExtractInput) -> WikiExtractOutput:
    # For section-structured content we use wikipedia-api
    api = wikipediaapi.Wikipedia(user_agent="LangGraphResearchAgent/1.0", language=args.lang)
    page = api.page(args.title)

    if not page.exists():
        # Try with search fallback if the exact title didn't exist
        _set_lang(args.lang)
        hits = wiki_search.search(args.title, results=1)
        if hits:
            page = api.page(hits[0])

    if not page.exists():
        return WikiExtractOutput(title=args.title, page_id=None, url=None, content=[])

    url = page.fullurl
    pid = str(page.pageid) if getattr(page, "pageid", None) else None

    # If no sections specified → return lead summary only
    if not args.sections:
        lead = page.summary or page.text.split("\n\n", 1)[0]
        return WikiExtractOutput(
            title=page.title,
            page_id=pid,
            url=url,
            content=[WikiSection(section="Lead", text=_truncate(lead, args.max_chars_per_section))],
        )

    # Otherwise match requested sections (case-insensitive, matches tail of full path)
    flat = _flatten_sections(page)
    wanted_lower = [s.lower() for s in args.sections]
    selected: List[WikiSection] = []

    for full_name, sec in flat:
        tail = full_name.split("/")[-1].lower()
        if any(tail == w or tail.endswith(w) for w in wanted_lower):
            txt = sec.text or ""
            if txt.strip():
                selected.append(WikiSection(section=sec.title, text=_truncate(txt, args.max_chars_per_section)))

    # If sections not found, gracefully fall back to lead
    if not selected:
        lead = page.summary or page.text.split("\n\n", 1)[0]
        selected = [WikiSection(section="Lead", text=_truncate(lead, args.max_chars_per_section))]

    return WikiExtractOutput(
        title=page.title, page_id=pid, url=url, content=selected
    )


def _extract_tool(**kwargs) -> WikiExtractOutput:
    args = WikiExtractInput(**kwargs)
    return _extract_impl(args)



wikipedia_extract_tool = StructuredTool.from_function(
    name="wikipedia_extract",
    description=(
        "Extract specific sections (or the lead summary) from a Wikipedia page. "
        "Use after wikipedia_lookup when you need concrete details from particular sections."
    ),
    func=_extract_tool,
    args_schema=WikiExtractInput,
    return_schema=WikiExtractOutput,
)
