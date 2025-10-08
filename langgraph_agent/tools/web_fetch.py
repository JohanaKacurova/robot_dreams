# tools/web_fetch.py
from __future__ import annotations

import os
import time
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import requests
from pydantic import BaseModel, Field, HttpUrl
from langchain_core.tools import StructuredTool

# Optional extractors
try:
    import trafilatura  # robust article extraction
except Exception:
    trafilatura = None

try:
    from bs4 import BeautifulSoup  # fallback for HTML
except Exception:
    BeautifulSoup = None

try:
    from pypdf import PdfReader  # PDF text
except Exception:
    PdfReader = None


# =========================
# Schemas
# =========================
class WebFetchInput(BaseModel):
    url: HttpUrl = Field(..., description="HTTP/HTTPS URL to fetch.")
    max_chars: int = Field(8000, ge=500, le=200000, description="Max characters of cleaned text to return.")
    timeout_sec: int = Field(12, ge=4, le=60, description="Network timeout for each request.")
    user_agent: Optional[str] = Field(
        default="Mozilla/5.0 (compatible; WebFetch/1.0; +https://example.local)",
        description="User-Agent header to send."
    )

class WebFetchOutput(BaseModel):
    url: HttpUrl
    final_url: str
    status_code: int
    content_type: str
    title: Optional[str] = None
    text: str
    chars_returned: int
    truncated: bool


# =========================
# Helpers
# =========================
def _clean(text: str) -> str:
    # Normalize whitespace, keep paragraphs
    if not text:
        return ""
    # Collapse very long spaces while keeping newlines
    text = text.replace("\r", "")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]  # drop empty lines
    return "\n\n".join(lines)


def _extract_html(html_bytes: bytes, base_url: str) -> tuple[str, Optional[str]]:
    title = None
    text = ""

    # 1) Try trafilatura
    if trafilatura is not None:
        try:
            # trafilatura works best with decoded strings; let it detect encoding
            html_str = html_bytes.decode("utf-8", errors="ignore")
            text = trafilatura.extract(
                html_str,
                include_comments=False,
                include_tables=False,
                include_formatting=False,
                no_fallback=True,
                favor_recall=False,
                url=base_url,
            ) or ""
            # Try to get a title via metadata
            try:
                meta = trafilatura.metadata.extract_metadata(html_str, url=base_url)  # may be None
                if meta and getattr(meta, "title", None):
                    title = meta.title
            except Exception:
                pass
        except Exception:
            text = ""

    # 2) Fallback to BeautifulSoup if needed
    if (not text) and (BeautifulSoup is not None):
        try:
            soup = BeautifulSoup(html_bytes, "html.parser")
            # Title
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            # Heuristic: remove scripts/styles
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            raw = soup.get_text("\n")
            text = raw or ""
        except Exception:
            text = ""

    return _clean(text), title


def _extract_pdf(pdf_bytes: bytes) -> tuple[str, Optional[str]]:
    if PdfReader is None:
        return "", None
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        title = None
        try:
            meta = reader.metadata or {}
            title = getattr(meta, "title", None) or meta.get("/Title")
            if title:
                title = str(title).strip()
        except Exception:
            pass

        # Concatenate first N pages (keep it reasonable)
        MAX_PAGES = 20
        pages = min(len(reader.pages), MAX_PAGES)
        parts = []
        for i in range(pages):
            try:
                parts.append(reader.pages[i].extract_text() or "")
            except Exception:
                continue
        text = "\n\n".join(parts)
        return _clean(text), title
    except Exception:
        return "", None


def _detect_is_pdf(url: str, content_type: str) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    # URL heuristic
    return url.lower().endswith(".pdf")


# =========================
# Core implementation
# =========================
def _web_fetch_impl(args: WebFetchInput) -> WebFetchOutput:
    headers = {"User-Agent": args.user_agent or "WebFetch/1.0"}
    t0 = time.time()

    # Single GET with redirects
    resp = requests.get(
        str(args.url),
        headers=headers,
        timeout=args.timeout_sec,
        allow_redirects=True,
        stream=True,
    )
    final_url = str(resp.url)
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

    # Read up to a sane cap for memory; PDFs can be big
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    content = b""
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if chunk:
            content += chunk
            if len(content) > MAX_BYTES:
                break

    # Decide extractor
    if _detect_is_pdf(final_url, ctype):
        text, title = _extract_pdf(content)
        ctype = ctype or "application/pdf"
    else:
        text, title = _extract_html(content, final_url)
        ctype = ctype or "text/html"

    # Truncate
    truncated = False
    if len(text) > args.max_chars:
        text = text[: args.max_chars].rstrip()
        truncated = True

    # Best-effort title
    if (not title) and text:
        # first non-empty line as title fallback
        title = text.split("\n", 1)[0][:140]

    _ = (time.time() - t0)  # latency unused in output (StructuredTool schema above)

    return WebFetchOutput(
        url=args.url,
        final_url=final_url,
        status_code=resp.status_code,
        content_type=ctype or "application/octet-stream",
        title=title,
        text=text,
        chars_returned=len(text),
        truncated=truncated,
    )


# =========================
# LangChain StructuredTool wrapper
# =========================
def _web_fetch(**kwargs) -> WebFetchOutput:
    args = WebFetchInput(**kwargs)
    return _web_fetch_impl(args)

web_fetch_tool = StructuredTool.from_function(
    name="web_fetch",
    description=(
        "Fetch a web page or PDF and return cleaned text and metadata. "
        "Use after web_search when you need the full content for summarization/citation."
    ),
    func=_web_fetch,
    args_schema=WebFetchInput,
    return_schema=WebFetchOutput,
)
