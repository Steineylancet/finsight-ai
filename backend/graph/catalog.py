"""
FinSight AI — Department catalog and resolver.

The canonical department list is the whitelist every tool validates against.
LLMs (and users) say "IT", "HR" or "Marketing"; the data is keyed on
"IT Infrastructure", "Human Resources" and three regional Marketing teams.
resolve_department() maps free text to exactly one canonical name, or reports
why it can't (unknown, or ambiguous across regions) so the caller can ask.
"""

import difflib
import re
from dataclasses import dataclass, field

DEPARTMENTS: list[str] = [
    "Customer Success - APAC",
    "Customer Success - Americas",
    "Customer Success - EMEA",
    "Cybersecurity",
    "Data & Analytics",
    "Executive",
    "Facilities & Real Estate",
    "Field Operations - APAC",
    "Field Operations - Americas",
    "Field Operations - EMEA",
    "Finance",
    "Finance Operations",
    "Human Resources",
    "IT Infrastructure",
    "Legal",
    "Marketing - APAC",
    "Marketing - Americas",
    "Marketing - EMEA",
    "Procurement",
    "Product Management",
    "Risk & Compliance",
    "Sales - APAC",
    "Sales - Americas",
    "Sales - EMEA",
    "Software Engineering",
    "Supply Chain",
]

_REGIONAL_GROUPS = {
    "customer success": "Customer Success",
    "cs": "Customer Success",
    "field operations": "Field Operations",
    "field ops": "Field Operations",
    "marketing": "Marketing",
    "sales": "Sales",
}

_REGION_ALIASES = {
    "APAC": ["apac", "asia", "asia pacific", "asia-pacific", "pacific", "india", "singapore", "australia", "japan"],
    "Americas": ["americas", "america", "us", "usa", "na", "north america", "latam", "canada"],
    "EMEA": ["emea", "europe", "eu", "uk", "middle east", "africa"],
}

_ALIASES = {
    "it": "IT Infrastructure",
    "it infra": "IT Infrastructure",
    "infrastructure": "IT Infrastructure",
    "eng": "Software Engineering",
    "engineering": "Software Engineering",
    "software eng": "Software Engineering",
    "swe": "Software Engineering",
    "hr": "Human Resources",
    "people": "Human Resources",
    "d and a": "Data & Analytics",
    "data": "Data & Analytics",
    "analytics": "Data & Analytics",
    "data and analytics": "Data & Analytics",
    "facilities": "Facilities & Real Estate",
    "real estate": "Facilities & Real Estate",
    "security": "Cybersecurity",
    "cyber": "Cybersecurity",
    "infosec": "Cybersecurity",
    "product": "Product Management",
    "pm": "Product Management",
    "finance ops": "Finance Operations",
    "finops": "Finance Operations",
    "risk": "Risk & Compliance",
    "compliance": "Risk & Compliance",
    "exec": "Executive",
    "executive office": "Executive",
}


def _norm(text: str) -> str:
    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(dept|department|team|org|function)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_CANONICAL_BY_NORM = {_norm(d): d for d in DEPARTMENTS}


@dataclass
class Resolution:
    department: str | None
    candidates: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.department is not None

    def message(self, raw: str) -> str:
        if self.candidates:
            options = ", ".join(self.candidates)
            return f'"{raw}" matches more than one department — did you mean one of: {options}?'
        return (
            f'"{raw}" is not a department in the planning data. '
            f"Valid departments are: {', '.join(DEPARTMENTS)}."
        )


def _detect_region(norm_text: str) -> str | None:
    for region, words in _REGION_ALIASES.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", norm_text):
                return region
    return None


def resolve_department(raw: str | None) -> Resolution:
    if not raw or not str(raw).strip():
        return Resolution(None)

    n = _norm(raw)
    if n in _CANONICAL_BY_NORM:
        return Resolution(_CANONICAL_BY_NORM[n])

    region = _detect_region(n)
    for alias, group in sorted(_REGIONAL_GROUPS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", n):
            members = [d for d in DEPARTMENTS if d.startswith(group + " - ")]
            if region:
                return Resolution(f"{group} - {region}")
            return Resolution(None, members)

    if n in _ALIASES:
        return Resolution(_ALIASES[n])

    close = difflib.get_close_matches(n, list(_CANONICAL_BY_NORM), n=3, cutoff=0.8)
    if len(close) == 1:
        return Resolution(_CANONICAL_BY_NORM[close[0]])
    if close:
        return Resolution(None, [_CANONICAL_BY_NORM[c] for c in close])
    return Resolution(None)
