"""Parse Sports-Reference-style HTML tables (BBRef, SR CBB).

These sites mark every cell with `data-stat`, and hide many tables inside HTML
comments to defer rendering. We read both visible and commented-out tables.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag


def _soup(html: bytes | str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def find_table(html: bytes | str, table_id: str) -> Tag | None:
    soup = _soup(html)
    table = soup.find("table", id=table_id)
    if isinstance(table, Tag):
        return table
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if table_id in comment:
            inner = _soup(str(comment)).find("table", id=table_id)
            if isinstance(inner, Tag):
                return inner
    return None


def table_ids(html: bytes | str) -> list[str]:
    """All table ids on the page, including commented-out ones."""
    soup = _soup(html)
    ids = [str(t.get("id")) for t in soup.find_all("table") if t.get("id")]
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if "<table" in comment:
            ids += [str(t.get("id")) for t in _soup(str(comment)).find_all("table") if t.get("id")]
    return ids


def parse_table(html: bytes | str, table_id: str, *, body_only: bool = True) -> pd.DataFrame:
    """Return one row per data row, keyed by `data-stat`.

    For each linked cell, an extra `<stat>__href` column holds the link. For cells
    with `data-append-csv` (player IDs on BBRef), `<stat>__id` holds that value.
    Header rows repeated inside the body are skipped. Values are left as strings.
    """
    table = find_table(html, table_id)
    if table is None:
        return pd.DataFrame()
    container = table.find("tbody") if body_only else table
    if not isinstance(container, Tag):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tr in container.find_all("tr"):
        classes = tr.get("class") or []
        if any(c in classes for c in ("thead", "over_header", "spacer")):
            continue
        row: dict[str, Any] = {}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if not stat:
                continue
            row[str(stat)] = cell.get_text(strip=True)
            link = cell.find("a")
            if isinstance(link, Tag) and link.get("href"):
                row[f"{stat}__href"] = link["href"]
            if cell.get("data-append-csv"):
                row[f"{stat}__id"] = cell["data-append-csv"]
        if row:
            rows.append(row)
    return pd.DataFrame(rows)
