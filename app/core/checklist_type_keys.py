"""
Canonical checklist type keys used in Supabase (`checklist_types.type_key`).

MVP: exactly three product-facing frameworks. Workbook filenames map onto these
so ingestion and analysis use the same keys the UI sends.
"""

from __future__ import annotations

from pathlib import Path

# Keys must match rows in `public.checklist_types` (and all `checklist_items`).
CANONICAL_KEYS: tuple[str, ...] = ("ifrs", "frs102", "frs105")

DISPLAY_NAME_BY_KEY: dict[str, str] = {
    "ifrs": "IFRS",
    "frs102": "FRS 102",
    "frs105": "FRS 105",
}

# Known XLSX stems (lowercase) -> canonical key. Keeps one checklist per product type.
_WORKBOOK_STEM_TO_KEY: dict[str, str] = {
    "ifrs_dc_2025": "ifrs",
    "listed_co_dc_2025": "ifrs",
    "frs1021a_dc_2025": "frs102",
    "llp_frs102_dc_2025": "frs102",
    "private_co_frs102_2025": "frs102",
    "charities_frs102_dc_2025": "frs102",
    "frs105_dc_2025": "frs105",
}


def normalize_stem(stem: str) -> str:
    key = stem.lower().replace("&", "and").replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def type_key_from_workbook_path(path: str | Path) -> str:
    """Map a checklist workbook filename to a canonical type_key."""
    stem = normalize_stem(Path(path).stem)
    if stem in _WORKBOOK_STEM_TO_KEY:
        return _WORKBOOK_STEM_TO_KEY[stem]
    if "frs105" in stem:
        return "frs105"
    if "listed_co" in stem or stem.startswith("ifrs") or "ifrs" in stem:
        return "ifrs"
    if "frs102" in stem or "1021a" in stem or ("charities" in stem and "frs" in stem):
        return "frs102"
    return stem


def resolve_framework_form_value(framework: str) -> str:
    """
    Map UI / API `framework` form field to canonical type_key.

    Accepts display labels, canonical keys, and legacy workbook-style names.
    """
    raw = framework.strip()
    if not raw:
        raise ValueError("framework is required")

    compact = raw.lower().replace(" ", "").replace("-", "_")

    aliases: dict[str, str] = {
        "ifrs": "ifrs",
        "ifrs_dc_2025": "ifrs",
        "frs102": "frs102",
        "frs_102": "frs102",
        "frs1021a": "frs102",
        "frs1021a_dc_2025": "frs102",
        "frs105": "frs105",
        "frs_105": "frs105",
        "frs105_dc_2025": "frs105",
        "charities_frs102_dc_2025": "frs102",
        "charitiesfrs102": "frs102",
        "listed_co_dc_2025": "ifrs",
        "llp_frs102_dc_2025": "frs102",
        "private_co_frs102_2025": "frs102",
    }

    if compact in aliases:
        return aliases[compact]
    if compact in CANONICAL_KEYS:
        return compact

    raise ValueError(
        f"Unknown framework '{framework}'. Use one of: IFRS, FRS 102, FRS 105 "
        f"(or keys: {', '.join(CANONICAL_KEYS)})."
    )


def display_name_for_key(type_key: str) -> str:
    return DISPLAY_NAME_BY_KEY.get(type_key, type_key.replace("_", " ").upper())
