"""Web fetch tool with markdown extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.tools.registry import GLOBAL_REGISTRY


@dataclass
class WebContent:
    """Structured result of a web fetch."""

    url: str
    status: int
    final_url: str
    content_type: str
    body: str
    title: str | None = None
    extract_mode: str = "markdown"


def _html_to_markdown(html: str) -> str:
    """Convert HTML to clean markdown via ``markdownify``."""
    try:
        from markdownify import markdownify as md
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("markdownify is required for markdown extraction") from exc
    return md(html, heading_style="ATX", strip=["script", "style"]).strip()


def _html_to_structured(html: str) -> str:
    """Extract the main article body via ``trafilatura``."""
    try:
        import trafilatura
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("trafilatura is required for structured extraction") from exc
    out = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    return out.strip()


@GLOBAL_REGISTRY.tool(
    description=(
        "Fetch a URL and convert the response. extract_mode='markdown' converts HTML to clean "
        "markdown, 'raw' returns the raw body, 'structured' uses trafilatura article extraction."
    ),
    side_effect="network",
    timeout=45.0,
)
async def web_fetch(
    url: str,
    extract_mode: Literal["markdown", "raw", "structured"] = "markdown",
    timeout: int = 30,
) -> dict:
    """Fetch ``url`` and return a structured dict.

    Args:
        url: The URL to fetch.
        extract_mode: How to post-process the body.
        timeout: HTTP timeout in seconds.
    """
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url)
        body = response.text
        ctype = response.headers.get("content-type", "")
    title: str | None = None
    if "html" in ctype:
        if extract_mode == "markdown":
            body = _html_to_markdown(body)
        elif extract_mode == "structured":
            body = _html_to_structured(body)
        import re

        m = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()
    elif extract_mode != "raw":
        body = body
    return {
        "url": url,
        "status": response.status_code,
        "final_url": str(response.url),
        "content_type": ctype,
        "title": title,
        "body": body[:200_000],
        "extract_mode": extract_mode,
    }
