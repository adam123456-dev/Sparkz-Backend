from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.checklists import ChecklistItem, parse_workbook
from app.checklists.llm_keywords import generate_rule_keywords_with_openai
from app.checklists.retrieval_embedding import retrieval_embedding_source_text
from app.core.checklist_type_keys import display_name_for_key, type_key_from_workbook_path

_SUPABASE_CLIENT = None
_SETTINGS = None
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse checklist XLSX files and upsert normalized rows into Supabase."
    )
    parser.add_argument(
        "--workbook",
        action="append",
        dest="workbooks",
        help="Absolute or relative workbook path. Repeat for multiple files.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=REPO_ROOT,
        help="Root directory to scan recursively for .xlsx files when --workbook is omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Supabase upsert batch size (default: 250).",
    )
    parser.add_argument(
        "--replace-type",
        action="store_true",
        help="Delete existing checklist rows for each type_key before inserting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and summarize but do not write to Supabase.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Sync checklist rows only, skip embedding generation.",
    )
    parser.add_argument(
        "--only-embeddings",
        action="store_true",
        help="Do not sync rows; only generate embeddings for existing checklist_items.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=50,
        help="OpenAI embedding batch size (default: 50).",
    )
    parser.add_argument(
        "--refresh-embeddings",
        action="store_true",
        help="Re-embed all checklist rows for each type (obligation-only vectors). "
        "Use after upgrading embedding strategy; requires OPENAI_API_KEY.",
    )
    return parser.parse_args()


def item_key_for(type_key: str, item: ChecklistItem) -> str:
    raw = "||".join(
        [
            type_key,
            item.sheet_name.strip(),
            item.requirement_id.strip(),
            item.requirement_text.strip(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def discover_workbooks(workbooks: list[str] | None, scan_root: Path) -> list[Path]:
    if workbooks:
        found = [Path(path).resolve() for path in workbooks]
    else:
        found = sorted(scan_root.rglob("*.xlsx"))

    valid = [path for path in found if path.exists() and path.is_file()]
    if not valid:
        raise SystemExit("No workbook files found.")

    # Deduplicate by file name when mirrored folders exist.
    deduped: dict[str, Path] = {}
    for path in valid:
        file_name = path.name.lower()
        if file_name not in deduped:
            deduped[file_name] = path
            continue
        # Prefer the one under "Disclosure Checklists" when both exist.
        existing = deduped[file_name]
        if "disclosure checklists" in str(path).lower() and "disclosure checklists" not in str(existing).lower():
            deduped[file_name] = path
    return sorted(deduped.values())


def chunked(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]


def upsert_type(type_key: str, display_name: str, source_workbook: str) -> None:
    supabase = get_supabase_client_lazy()
    supabase.table("checklist_types").upsert(
        [
            {
                "type_key": type_key,
                "display_name": display_name,
                "source_workbook": source_workbook,
                "is_active": True,
            }
        ],
        on_conflict="type_key",
    ).execute()


def delete_items_for_type(type_key: str) -> int:
    supabase = get_supabase_client_lazy()
    response = (
        supabase.table("checklist_items")
        .delete()
        .eq("checklist_type_key", type_key)
        .execute()
    )
    return len(response.data or [])


def upsert_items(rows: list[dict[str, Any]], batch_size: int) -> int:
    supabase = get_supabase_client_lazy()
    deduped_rows = dedupe_rows_by_item_key(rows)
    total = 0
    for batch in chunked(deduped_rows, batch_size):
        supabase.table("checklist_items").upsert(batch, on_conflict="item_key").execute()
        total += len(batch)
    return total


def rows_for_workbook(path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    items = parse_workbook(path)
    if not items:
        return "", "", []

    type_key = type_key_from_workbook_path(path)
    display_name = display_name_for_key(type_key)
    rows: list[dict[str, Any]] = []

    settings = get_settings_lazy()
    total_items = len(items)
    for idx, item in enumerate(items, start=1):
        row = asdict(item)
        row["item_key"] = item_key_for(type_key, item)
        row["checklist_type_key"] = type_key
        row["embedding_text"] = item.embedding_text
        keywords, section_hints = _build_search_keywords_for_item(item, settings)
        row["search_keywords"] = keywords
        row["section_hints"] = section_hints
        rows.append(row)
        if idx % 20 == 0 or idx == total_items:
            print(f"  - keyword generation progress: {idx}/{total_items}", flush=True)

    return type_key, display_name, rows


def _build_search_keywords_for_item(item: ChecklistItem, settings: Any) -> tuple[list[str], list[str]]:
    """
    OpenAI-generated keyword tiers only.
    Stored ``search_keywords`` keeps strict terms first, then likely terms.
    """
    print("*****************")
    print("item", item)
    print("*****************")
    section_hints: list[str] = []
    key = (settings.openai_api_key or "").strip()
    if not key:
        return [], section_hints
    try:
        keywords = generate_rule_keywords_with_openai(
            api_key=key,
            model=settings.openai_chat_model,
            requirement_text=item.requirement_text,
            requirement_text_leaf=item.requirement_text_leaf,
            reference_text=item.reference_text,
        )
        return keywords, section_hints
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI keyword generation failed for %s: %s", item.requirement_id, exc)
    return [], section_hints


def dedupe_rows_by_item_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, str]] = {}
    for row in rows:
        # Last write wins for deterministic behavior if duplicates occur.
        deduped[row["item_key"]] = row
    return list(deduped.values())


def get_supabase_client_lazy():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        from app.db.supabase import get_supabase_client

        _SUPABASE_CLIENT = get_supabase_client()
    return _SUPABASE_CLIENT


def get_settings_lazy():
    global _SETTINGS
    if _SETTINGS is None:
        from app.core.config import get_settings

        _SETTINGS = get_settings()
    return _SETTINGS


def fetch_items_for_embedding(type_key: str, only_missing: bool = True) -> list[dict[str, str]]:
    supabase = get_supabase_client_lazy()
    rows: list[dict[str, str]] = []
    page_size = 1000
    offset = 0

    while True:
        query = (
            supabase.table("checklist_items")
            .select("item_key,checklist_type_key,requirement_text,requirement_text_leaf")
            .eq("checklist_type_key", type_key)
            .range(offset, offset + page_size - 1)
        )
        response = query.execute()
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not only_missing:
        return rows

    embedded_keys: set[str] = set()
    offset = 0
    while True:
        response = (
            supabase.table("checklist_item_embeddings")
            .select("item_key")
            .eq("checklist_type_key", type_key)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        embedded_keys.update(str(item["item_key"]) for item in batch if "item_key" in item)
        if len(batch) < page_size:
            break
        offset += page_size

    return [row for row in rows if row["item_key"] not in embedded_keys]


def build_openai_embeddings(texts: list[str]) -> list[list[float]]:
    settings = get_settings_lazy()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    payload = {
        "model": settings.openai_embedding_model,
        "input": texts,
    }
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.openai_api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:  # nosec B310
        body = json.loads(response.read().decode("utf-8"))
    data = body.get("data", [])
    return [item["embedding"] for item in data]


def upsert_embeddings(type_key: str, batch_rows: list[dict[str, str]], embeddings: list[list[float]]) -> int:
    if len(batch_rows) != len(embeddings):
        raise RuntimeError("Embedding count mismatch.")

    settings = get_settings_lazy()
    payload = []
    for row, embedding in zip(batch_rows, embeddings):
        payload.append(
            {
                "item_key": row["item_key"],
                "checklist_type_key": type_key,
                "model": settings.openai_embedding_model,
                "embedding": embedding,
            }
        )

    supabase = get_supabase_client_lazy()
    supabase.table("checklist_item_embeddings").upsert(payload, on_conflict="item_key").execute()
    return len(payload)


def generate_embeddings_for_type(
    type_key: str,
    embedding_batch_size: int,
    dry_run: bool,
    *,
    refresh_all: bool,
) -> int:
    candidate_rows = fetch_items_for_embedding(type_key, only_missing=not refresh_all)
    if not candidate_rows:
        print(f"  - embeddings up to date for type '{type_key}'")
        return 0

    print(f"  - embedding {len(candidate_rows)} rows for type '{type_key}'")
    if dry_run:
        return len(candidate_rows)

    total = 0
    for batch in chunked(candidate_rows, embedding_batch_size):
        texts = [
            retrieval_embedding_source_text(
                requirement_text=str(row.get("requirement_text") or ""),
                requirement_text_leaf=str(row.get("requirement_text_leaf") or ""),
            )
            for row in batch
        ]
        vectors = build_openai_embeddings(texts)
        total += upsert_embeddings(type_key, batch, vectors)
    return total


def main() -> None:
    args = parse_args()
    if args.only_embeddings and args.replace_type:
        raise SystemExit("--replace-type cannot be used with --only-embeddings.")

    workbooks = discover_workbooks(args.workbooks, args.scan_root)
    print(f"Found {len(workbooks)} workbook(s).")

    all_rows = 0
    type_keys_processed: list[str] = []
    for workbook in workbooks:
        type_key, display_name, rows = rows_for_workbook(workbook)
        if not rows:
            print(f"[SKIP] {workbook.name}: no extractable rows")
            continue

        print(f"[PARSED] {workbook.name}: {len(rows)} rows -> type_key '{type_key}'")
        deduped_count = len(dedupe_rows_by_item_key(rows))
        duplicate_count = len(rows) - deduped_count
        if duplicate_count:
            print(f"  - collapsed {duplicate_count} duplicate item_key rows before upsert")
        all_rows += len(rows)
        type_keys_processed.append(type_key)

        if args.only_embeddings:
            continue

        if args.dry_run:
            continue
        upsert_type(type_key, display_name, workbook.name)
        if args.replace_type:
            removed = delete_items_for_type(type_key)
            print(f"  - removed {removed} existing rows for type '{type_key}'")

        inserted = upsert_items(rows, args.batch_size)
        print(f"  - upserted {inserted} rows for type '{type_key}'")

    print(f"Done. Total parsed rows: {all_rows}")
    if args.skip_embeddings: 
        print("Embedding step skipped (--skip-embeddings).")
        return

    unique_type_keys = sorted(set(type_keys_processed))
    if not unique_type_keys:
        print("No types found for embedding step.")
        return

    total_embeddings = 0
    for type_key in unique_type_keys:
        embedded = generate_embeddings_for_type(
            type_key,
            args.embedding_batch_size,
            args.dry_run,
            refresh_all=args.refresh_embeddings,
        )
        if embedded:
            print(f"  - upserted {embedded} embeddings for type '{type_key}'")
        total_embeddings += embedded
    print(f"Embedding step complete. Total embeddings processed: {total_embeddings}")


if __name__ == "__main__":
    main()

