-- Sparkz checklist ingestion schema
-- Run this in Supabase SQL Editor before running ingestion scripts.

create extension if not exists vector;

create table if not exists public.checklist_types (
  id uuid primary key default gen_random_uuid(),
  type_key text not null unique,
  display_name text not null,
  source_workbook text not null,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.checklist_items (
  id uuid primary key default gen_random_uuid(),
  item_key text not null unique,
  checklist_type_key text not null references public.checklist_types(type_key) on delete cascade,
  source_workbook text not null,
  framework text not null,
  sheet_name text not null,
  section_path text not null default '',
  requirement_id text not null,
  requirement_text text not null,
  requirement_base_id text not null default '',
  clause_path text not null default '',
  requirement_text_leaf text not null default '',
  notes_text text not null default '',
  item_kind text not null default 'rule' check (item_kind in ('rule', 'group', 'note')),
  reference_text text not null default '',
  embedding_text text not null,
  -- Derived at sync time (not separate XLSX columns): lexical pre-filter + UI hints
  search_keywords text[] not null default '{}'::text[],
  section_hints text[] not null default '{}'::text[],
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_checklist_items_type on public.checklist_items(checklist_type_key);
create index if not exists idx_checklist_items_req_id on public.checklist_items(requirement_id);

create table if not exists public.checklist_item_embeddings (
  item_key text primary key references public.checklist_items(item_key) on delete cascade,
  checklist_type_key text not null references public.checklist_types(type_key) on delete cascade,
  model text not null,
  embedding vector(1536) not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_checklist_item_embeddings_type on public.checklist_item_embeddings(checklist_type_key);
create index if not exists idx_checklist_item_embeddings_vector
  on public.checklist_item_embeddings
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

drop function if exists public.match_checklist_items(vector(1536), text, int);

create or replace function public.match_checklist_items (
  query_embedding vector(1536),
  filter_type_key text,
  match_count int default 8
)
returns table (
  item_key text,
  checklist_type_key text,
  requirement_id text,
  requirement_base_id text,
  clause_path text,
  requirement_text text,
  requirement_text_leaf text,
  reference_text text,
  sheet_name text,
  section_path text,
  search_keywords text[],
  section_hints text[],
  similarity float
)
language sql
as $$
  select
    ci.item_key,
    ci.checklist_type_key,
    ci.requirement_id,
    ci.requirement_base_id,
    ci.clause_path,
    ci.requirement_text,
    ci.requirement_text_leaf,
    ci.reference_text,
    ci.sheet_name,
    ci.section_path,
    ci.search_keywords,
    ci.section_hints,
    1 - (ce.embedding <=> query_embedding) as similarity
  from public.checklist_item_embeddings ce
  join public.checklist_items ci on ci.item_key = ce.item_key
  where ci.checklist_type_key = filter_type_key
  order by ce.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

