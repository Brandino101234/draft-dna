"""Player-name normalization for cross-source matching."""

from __future__ import annotations

import re

from unidecode import unidecode

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Common nickname/spelling variants seen across sources (normalized form -> canonical).
ALIASES = {
    "nene hilario": "nene",
    "ron artest": "metta world peace",
    "metta sandiford artest": "metta world peace",
    "enes freedom": "enes kanter",
    "enes kanter freedom": "enes kanter",
}


def normalize_name(name: str | None, *, drop_suffix: bool = True) -> str:
    """Lowercase ASCII, punctuation removed, suffixes (Jr., III) optionally dropped.

    'Luka Dončić' -> 'luka doncic'; 'Kevin Porter Jr.' -> 'kevin porter';
    "D'Angelo Russell" -> 'dangelo russell'; 'P.J. Tucker' -> 'pj tucker'.
    """
    if not name:
        return ""
    s = unidecode(name).lower()
    s = re.sub(r"[.'`\u2019]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = s.split()
    if drop_suffix:
        tokens = [t for t in tokens if t not in SUFFIXES]
    out = " ".join(tokens)
    return ALIASES.get(out, out)


def last_name(name: str | None) -> str:
    parts = normalize_name(name).split()
    return parts[-1] if parts else ""
