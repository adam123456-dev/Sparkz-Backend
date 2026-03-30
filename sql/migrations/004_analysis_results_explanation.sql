-- Legacy patch: explanation and evidence columns now live in
-- 002_analysis_pipeline_schema.sql for fresh installs.
-- Use this only when upgrading an older database that ran 001+002 before those columns existed.

alter table public.analysis_results
  add column if not exists explanation text;

alter table public.analysis_results
  add column if not exists evidence jsonb;
