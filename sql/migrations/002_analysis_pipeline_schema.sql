-- Analysis pipeline schema for uploaded financial statements

create table if not exists public.analyses (
  id uuid primary key default gen_random_uuid(),
  company_name text not null,
  checklist_type_key text not null references public.checklist_types(type_key),
  status text not null default 'queued',
  progress int not null default 0,
  message text not null default 'Queued for processing.',
  steps jsonb not null default '[]'::jsonb,
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_analyses_status on public.analyses(status);
create index if not exists idx_analyses_checklist_type on public.analyses(checklist_type_key);

create table if not exists public.analysis_documents (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  original_filename text not null,
  storage_path text not null,
  page_count int not null default 0,
  created_at timestamptz not null default now()
);

create table if not exists public.analysis_chunks (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  chunk_index int not null,
  page_number int not null,
  text_redacted text not null,
  text_hash text not null,
  heading_guess text not null default '',
  search_tsv tsvector generated always as (to_tsvector('english', coalesce(text_redacted, ''))) stored,
  created_at timestamptz not null default now(),
  unique (analysis_id, chunk_index)
);

create index if not exists idx_analysis_chunks_analysis on public.analysis_chunks(analysis_id);
create index if not exists idx_analysis_chunks_search_tsv on public.analysis_chunks using gin (search_tsv);

create table if not exists public.analysis_chunk_embeddings (
  chunk_id uuid primary key references public.analysis_chunks(id) on delete cascade,
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  model text not null,
  embedding vector(1536) not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_analysis_chunk_embeddings_analysis on public.analysis_chunk_embeddings(analysis_id);
create index if not exists idx_analysis_chunk_embeddings_vector
  on public.analysis_chunk_embeddings
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

create table if not exists public.analysis_results (
  id uuid primary key default gen_random_uuid(),
  analysis_id uuid not null references public.analyses(id) on delete cascade,
  item_key text not null references public.checklist_items(item_key) on delete cascade,
  status text not null,
  evidence_snippet text,
  evidence jsonb,
  explanation text,
  similarity float not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (analysis_id, item_key)
);

create index if not exists idx_analysis_results_analysis on public.analysis_results(analysis_id);
create index if not exists idx_analysis_results_status on public.analysis_results(status);

create or replace function public.match_analysis_chunks (
  query_embedding vector(1536),
  filter_analysis_id uuid,
  match_count int default 8
)
returns table (
  chunk_id uuid,
  page_number int,
  text_redacted text,
  similarity float
)
language sql
as $$
  select
    ac.id as chunk_id,
    ac.page_number,
    ac.text_redacted,
    1 - (ace.embedding <=> query_embedding) as similarity
  from public.analysis_chunk_embeddings ace
  join public.analysis_chunks ac on ac.id = ace.chunk_id
  where ace.analysis_id = filter_analysis_id
  order by ace.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

