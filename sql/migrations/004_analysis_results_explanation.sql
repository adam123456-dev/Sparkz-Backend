-- Short LLM rationale per requirement (token-capped in app).

alter table public.analysis_results
  add column if not exists explanation text;
