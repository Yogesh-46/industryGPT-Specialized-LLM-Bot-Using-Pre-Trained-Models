"""HTML cleaning utilities for Dataset A preprocessing."""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _first_main_node(soup: BeautifulSoup, selectors: list[str]) -> Tag | None:
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return None


def extract_title(html: str) -> str | None:
    """Extract document title from HTML ``<title>`` or first ``h1``."""
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    h1 = soup.find("h1")
    if h1:
        return " ".join(h1.get_text(" ", strip=True).split())
    return None


def clean_html(
    html: str,
    *,
    remove_tags: list[str] | None = None,
    main_content_selectors: list[str] | None = None,
    collapse_whitespace: bool = True,
    normalize_newlines: bool = True,
    strip_lines: bool = True,
    drop_empty_lines: bool = True,
) -> dict[str, Any]:
    """Clean HTML into plain text suitable for RAG corpus construction.

    Returns a dict with ``text``, ``title``, ``parser_ok``, and ``used_selector``.
    """
    remove_tags = remove_tags or []
    main_content_selectors = main_content_selectors or []

    try:
        soup = BeautifulSoup(html, "lxml")
        parser_ok = True
    except Exception:  # noqa: BLE001
        return {
            "text": "",
            "title": None,
            "parser_ok": False,
            "used_selector": None,
            "error": "BeautifulSoup parse failed",
        }

    if not soup.find():
        return {
            "text": "",
            "title": None,
            "parser_ok": False,
            "used_selector": None,
            "error": "Empty or malformed HTML",
        }

    title = None
    if soup.title and soup.title.get_text(strip=True):
        title = " ".join(soup.title.get_text(" ", strip=True).split())

    for tag_name in remove_tags:
        for node in soup.find_all(tag_name):
            node.decompose()

    # Remove common hidden / navigation helpers
    for node in soup.select("[hidden], .nav, .navbar, .sidebar, .toc, .breadcrumb, .breadcrumbs"):
        node.decompose()

    used_selector: str | None = None
    root: Tag | BeautifulSoup = soup
    main_node = _first_main_node(soup, main_content_selectors)
    if main_node is not None:
        root = main_node
        # Find which selector matched for provenance/debug
        for selector in main_content_selectors:
            if soup.select_one(selector) is main_node:
                used_selector = selector
                break

    if not title:
        h1 = root.find("h1") if isinstance(root, Tag) else soup.find("h1")
        if h1:
            title = " ".join(h1.get_text(" ", strip=True).split())

    # Prefer block-aware text extraction
    parts: list[str] = []
    for element in root.descendants:
        if isinstance(element, NavigableString):
            parent = element.parent
            if parent is None or parent.name in {"script", "style"}:
                continue
            text = str(element)
            if text.strip():
                parts.append(text)

    # Fallback to get_text which is usually cleaner for docs sites
    text = root.get_text("\n", strip=False)

    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        if collapse_whitespace:
            line = _WHITESPACE_RE.sub(" ", line)
        if strip_lines:
            line = line.strip()
        if drop_empty_lines and not line:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    if normalize_newlines:
        text = _MULTI_NEWLINE_RE.sub("\n\n", text).strip()

    return {
        "text": text,
        "title": title,
        "parser_ok": parser_ok,
        "used_selector": used_selector,
        "error": None,
    }
