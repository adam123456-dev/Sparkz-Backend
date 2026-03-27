"""Fetch requirement text for checklist items (for LLM prompts)."""

from __future__ import annotations

from app.db.supabase import get_supabase_client


def requirement_text_by_item_key(checklist_type_key: str, item_keys: list[str]) -> dict[str, str]:
    if not item_keys:
        return {}

    supabase = get_supabase_client()
    result: dict[str, str] = {}
    batch_size = 200
    for start in range(0, len(item_keys), batch_size):
        batch = item_keys[start : start + batch_size]
        rows = (
            supabase.table("checklist_items")
            .select("item_key,requirement_text")
            .eq("checklist_type_key", checklist_type_key)
            .in_("item_key", batch)
            .execute()
            .data
            or []
        )
        for row in rows:
            result[str(row["item_key"])] = str(row.get("requirement_text") or "")
    return result