"""Were draft-night pick trades worth it? (D042)

Every trade that moved a drafted player before his first season, from Basketball-Reference
transaction logs (cached player pages, no new requests). Each trade is split into what each
side gave: the traded player plus anyone listed "with" him goes one way; everything after
"for" comes back. Future picks are credited with the player they became ("... was later
selected"). Three-team deals and trades with pick-swap rights are not split and are excluded.

Value of an asset = its best 3-season stretch (season value incl. playoffs) from the season
after the trade on, so a veteran only counts for what he did after being dealt. Unknown
assets (players drafted before 1996, unresolved future picks) make a trade "incomplete".

Verdict, from the side that traded for the best pick in the deal ("buyer"): worth it if it
got > EVEN more value than it gave, not worth it if < -EVEN, otherwise even.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag

from draft_dna.config import Settings
from draft_dna.ingest import bbref
from draft_dna.ingest.fetcher import Fetcher, NotFoundError
from draft_dna.ingest.storage import read_table, write_table
from draft_dna.logging_utils import get_logger

log = get_logger(__name__)

EVEN = 0.25
PLAYER_HREF = re.compile(r"/players/\w/([\w.]+)\.html")
PICK = re.compile(r"draft pick", re.I)
LATER = re.compile(r"was later selected", re.I)


def _tokens(node: Tag) -> list[tuple[str, str]]:
    """Flatten a transaction into ('text', s) / ('player', id) / ('team', code) tokens."""
    out: list[tuple[str, str]] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            out.append(("text", str(child)))
        elif isinstance(child, Tag) and child.name == "a":
            m = PLAYER_HREF.search(str(child.get("href", "")))
            if m:
                out.append(("player", m.group(1)))
            elif child.get("data-attr-to") or child.get("data-attr-from"):
                out.append(("team", str(child.get("data-attr-to") or child.get("data-attr-from"))))
            else:
                out.append(("text", child.get_text()))
        elif isinstance(child, Tag):
            out.extend(_tokens(child))
    return out


def parse_trades(html: bytes, subject: str) -> list[dict[str, Any]]:
    raw = html.decode("utf-8", "ignore")
    start = raw.find('id="div_transactions"')
    if start < 0:
        return []
    soup = BeautifulSoup("<div " + raw[start : raw.find("</div>", start) + 6], "lxml")
    rows = []
    for p in soup.find_all("p", class_="transaction"):
        text = p.get_text(" ", strip=True)
        if not re.search(r"\btraded by\b", text, re.I):
            continue
        date_tag = p.find("strong")
        frm = p.find(attrs={"data-attr-from": True})
        to = p.find(attrs={"data-attr-to": True})
        # Three-team deals and pick-swap rights can't be split into two clean sides.
        multi = bool(re.search(r"\d-team trade|\bswap\b", text, re.I))
        toks = _tokens(p)
        # Split at the first " for " that follows the receiving team.
        seen_to, split = False, None
        for i, (kind, val) in enumerate(toks):
            if kind == "team" and isinstance(to, Tag) and val == to.get("data-attr-to"):
                seen_to = True
            if seen_to and kind == "text" and re.search(r"\bfor\b", val):
                split = i
                break
        sent = [subject] + [v for k, v in (toks[:split] if split else toks) if k == "player"]
        got = [v for k, v in toks[split:] if k == "player"] if split is not None else []
        tail = "".join(v for k, v in toks[split:] if k == "text") if split is not None else ""
        head = "".join(v for k, v in toks[:split] if k == "text") if split is not None else text
        rows.append(
            {
                "date": pd.to_datetime(date_tag.get_text(strip=True), errors="coerce")
                if isinstance(date_tag, Tag)
                else pd.NaT,
                "team_from": frm.get("data-attr-from") if isinstance(frm, Tag) else None,
                "team_to": to.get("data-attr-to") if isinstance(to, Tag) else None,
                "multi_team": multi,
                "sent": sorted(set(sent)),
                "received": sorted(set(got)),
                # picks not (yet) resolved to a player, and cash, on each side
                "sent_unresolved": max(0, len(PICK.findall(head)) - len(LATER.findall(head))),
                "received_unresolved": max(0, len(PICK.findall(tail)) - len(LATER.findall(tail))),
                "has_return": split is not None,
                "text": text[:400],
            }
        )
    return rows


def _post_trade_peak(values: pd.DataFrame, pid: str, first_season: int) -> float:
    v = values.loc[values.index == pid]
    v = v[v["season"] >= first_season].sort_values("season")["value"].to_numpy()
    if len(v) == 0:
        return 0.0
    best = -np.inf
    for i in range(len(v)):
        best = max(best, v[max(0, i - 2) : i + 1].sum() / 3)
    return float(best)


def build(s: Settings) -> pd.DataFrame:
    players = read_table("modeled", "core", "players", s).set_index("bbref_id")
    traded = players.index[players["drafted"] & players["traded_before_debut"].fillna(False)]
    tx = read_table("staging", "bbref", "player_transactions", s)
    drafted_on = tx[tx["kind"] == "drafted"].groupby("bbref_id")["date"].min()
    f = Fetcher(bbref.SOURCE, settings=s)
    records = []
    for pid in traded:
        try:
            html = f.get(bbref.player_url(pid))
        except NotFoundError:
            continue
        d0 = drafted_on.get(pid)
        if d0 is None or pd.isna(d0):
            continue
        end = d0 + pd.Timedelta(days=30) if d0.month >= 10 else pd.Timestamp(d0.year, 10, 1)
        for r in parse_trades(html, pid):
            if pd.notna(r["date"]) and d0 <= r["date"] < end:
                records.append(
                    {**r, "subject": pid, "draft_year": int(players.loc[pid, "draft_year"])}
                )
    if f.network_requests:
        log.warning("pick trades made %d network requests", f.network_requests)
    t = pd.DataFrame(records)
    # One row per trade: the same deal appears on every involved player's page.
    t["key"] = [
        (str(dt.date()), tuple(sorted({a, b})), tuple(sorted(set(x) | set(y))))
        for dt, a, b, x, y in zip(
            t["date"], t["team_from"], t["team_to"], t["sent"], t["received"], strict=True
        )
    ]
    t = t.drop_duplicates("key").reset_index(drop=True)

    vals = read_table("modeled", "outcomes", "season_values", s)
    vals = vals.assign(value=vals["value_blend"] + vals["value_playoff"]).set_index("bbref_id")
    known = set(players.index)
    pick_of = players["pick_overall"]
    rows = []
    for _, r in t.iterrows():
        first = int(r["date"].year) + (1 if r["date"].month >= 6 else 0)
        sides = {"sent": r["sent"], "received": r["received"]}
        val = {
            k: sum(_post_trade_peak(vals, p, first) for p in ids if p in known)
            for k, ids in sides.items()
        }
        unknown = (
            any(p not in known for p in r["sent"] + r["received"])
            or r["sent_unresolved"] > 0
            or r["received_unresolved"] > 0
            or not r["has_return"]
        )

        # Buyer = the side that received the best-drafted player from this draft.
        year = r["draft_year"]

        def best_pick(ids: list[str], year: int = year) -> float:
            """Best (lowest) pick among this draft's players in `ids`."""
            same = [
                pick_of.get(p)
                for p in ids
                if p in known and players.loc[p, "draft_year"] == year and players.loc[p, "drafted"]
            ]
            same = [x for x in same if pd.notna(x)]
            return min(same) if same else np.inf

        # 'sent' went from team_from to team_to, so team_to received the 'sent' players.
        to_gets, from_gets = r["sent"], r["received"]
        if best_pick(to_gets) <= best_pick(from_gets):
            buyer, seller, got, gave = r["team_to"], r["team_from"], to_gets, from_gets
            v_got, v_gave = val["sent"], val["received"]
        else:
            buyer, seller, got, gave = r["team_from"], r["team_to"], from_gets, to_gets
            v_got, v_gave = val["received"], val["sent"]
        got_known = [p for p in got if p in known and pd.notna(pick_of.get(p))]
        top_id = min(got_known, key=lambda p: pick_of.get(p)) if got_known else None
        top = min(
            (pick_of.get(p) for p in got if p in known and pd.notna(pick_of.get(p))), default=np.nan
        )
        diff = v_got - v_gave
        status = (
            "complex (not scored)"
            if r["multi_team"]
            else "incomplete (unknown assets)"
            if unknown
            else "worth it"
            if diff > EVEN
            else "not worth it"
            if diff < -EVEN
            else "even"
        )
        rows.append(
            {
                "draft_year": r["draft_year"],
                "date": r["date"],
                "buyer": buyer,
                "seller": seller,
                "top_pick": top,
                "top_id": top_id,
                "got": ", ".join(players["player_name"].get(p, p) for p in got),
                "gave": ", ".join(players["player_name"].get(p, p) for p in gave) or "(picks/cash)",
                "got_ids": ",".join(got),
                "gave_ids": ",".join(gave),
                "value_got": v_got,
                "value_gave": v_gave,
                "diff": diff,
                "verdict": status,
                "in_progress": r["draft_year"] >= s.draft_classes.in_progress[0],
                "text": r["text"],
            }
        )
    out = pd.DataFrame(rows)
    log.info(
        "pick trades: %d trades, verdicts %s", len(out), out["verdict"].value_counts().to_dict()
    )
    return out


def run(s: Settings) -> None:
    write_table(build(s), "modeled", "grading", "pick_trades", s)
