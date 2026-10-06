-- Knowledge base for retrieval (replaces the old retrieval_index.npz file).
-- Dimension 768 matches the default embedding model BAAI/bge-base-en-v1.5.
-- A model with a different dimension (bge-large = 1024, MiniLM = 384) needs a
-- new migration that alters the column and rebuilds the index.

create extension if not exists vector;

create table if not exists public.facts (
  fact_id             text primary key,
  fact                text not null,
  fact_type           text not null default 'other',
  source_url          text,
  source_page_title   text,
  source_section      text,
  source_excerpt      text,
  confidence          text,
  crawled_at          timestamptz,
  retrieval_text      text,
  -- Hand-curated "answer cards" (data/knowledge/answer_cards.jsonl): when a
  -- keyword appears in the question the card is returned verbatim.
  answer_card_keywords text[],
  -- Facts that explicitly state the head office; get a fixed ranking boost.
  is_authoritative    boolean not null default false,
  -- Any additional fields from the source JSONL, kept so nothing is lost.
  extra               jsonb not null default '{}'::jsonb,
  -- sha256(embedding model + embedded text): lets the sync job re-embed only
  -- facts whose text (or model) changed.
  embedding_hash      text not null,
  embedding           vector(768) not null,
  updated_at          timestamptz not null default now()
);

-- Approximate nearest-neighbour search, cosine distance (vectors are
-- L2-normalized, so this ranks identically to the previous dot product).
create index if not exists facts_embedding_hnsw
  on public.facts using hnsw (embedding vector_cosine_ops);

-- Boost-candidate lookups.
create index if not exists facts_fact_type_idx on public.facts (fact_type);
create index if not exists facts_authoritative_idx on public.facts (fact_id) where is_authoritative;
create index if not exists facts_answer_cards_idx on public.facts (fact_id) where answer_card_keywords is not null;

-- Which embedding model the stored vectors belong to, when last synced, etc.
create table if not exists public.index_meta (
  key   text primary key,
  value text not null
);

-- The API connects with a server-side database role that bypasses RLS; the
-- public Supabase Data API (anon / authenticated) must never read or write
-- these tables. RLS on with no policies = deny by default.
alter table public.facts enable row level security;
alter table public.index_meta enable row level security;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on table public.facts, public.index_meta from anon;
  end if;
  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on table public.facts, public.index_meta from authenticated;
  end if;
end
$$;
