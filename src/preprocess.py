"""Conservative text preprocessing for business entity resolution."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


# Keep ordinary business words (company, group, holdings) intact. Only remove
# unambiguous terminal legal designators in the separate core-name field.
_LEGAL_SUFFIX_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("limited", "liability", "company"), ("private", "limited"), ("pvt", "ltd"),
    ("incorporated",), ("corporation",), ("limited",), ("company", "llc"),
    ("corp",), ("inc",), ("llc",), ("llp",), ("plc",), ("ltd",),
    ("gmbh",), ("sarl",), ("sas",), ("bv",), ("oy",),
)
_ADDRESS_TOKEN_ALIASES = {
    "apt": "apartment", "ave": "avenue", "blvd": "boulevard", "bldg": "building",
    "ctr": "center", "fl": "floor", "hwy": "highway", "ln": "lane", "pkwy": "parkway",
    "rd": "road", "sq": "square", "ste": "suite",
}


def _is_missing(value: Any) -> bool:
    if value is None or type(value).__name__ == "NAType":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def normalize_text(value: Any) -> str:
    """Normalize case, ampersands, punctuation, and whitespace without deleting tokens."""
    raw = _as_text(value)
    if not raw:
        return ""
    normalized = unicodedata.normalize("NFKC", raw).casefold().replace("&", " and ")
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_business_name(value: Any) -> str:
    """Return a normalized name while retaining legal suffix information."""
    return normalize_text(value)


def normalize_business_name_core(value: Any) -> str:
    """Return a secondary normalized name with terminal legal suffixes removed."""
    tokens = normalize_business_name(value).split()
    changed = True
    while changed:
        changed = False
        for suffix in _LEGAL_SUFFIX_PATTERNS:
            if len(tokens) > len(suffix) and tuple(tokens[-len(suffix):]) == suffix:
                del tokens[-len(suffix):]
                changed = True
                break
    return " ".join(tokens)


def normalize_address(value: Any) -> str:
    """Normalize an address while retaining numeric and postal-code tokens."""
    return " ".join(_ADDRESS_TOKEN_ALIASES.get(token, token) for token in normalize_text(value).split())


def normalize_country(value: Any) -> str:
    """Normalize an open-set country value; do not map it to a fixed vocabulary."""
    return normalize_text(value)


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one record and add, rather than replace, normalized fields."""
    result = dict(record)
    name = record.get("name", record.get("business_name"))
    address = record.get("address", record.get("business_address"))
    result["name_normalized"] = normalize_business_name(name)
    result["name_core_normalized"] = normalize_business_name_core(name)
    result["address_normalized"] = normalize_address(address)
    result["country_normalized"] = normalize_country(record.get("country"))
    return result


def preprocess_dataframe(dataframe: Any) -> Any:
    """Return a copied DataFrame with additive normalized columns."""
    result = dataframe.copy()
    name_column = "name" if "name" in result.columns else "business_name"
    address_column = "address" if "address" in result.columns else "business_address"
    missing = {column for column in (name_column, address_column, "country") if column not in result.columns}
    if missing:
        raise KeyError(f"Missing required preprocessing columns: {sorted(missing)}")
    result["name_normalized"] = result[name_column].map(normalize_business_name)
    result["name_core_normalized"] = result[name_column].map(normalize_business_name_core)
    result["address_normalized"] = result[address_column].map(normalize_address)
    result["country_normalized"] = result["country"].map(normalize_country)
    return result
