create table if not exists meta_pending_connections (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null unique,
  pages jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists meta_pending_connections_owner_id_idx on meta_pending_connections (owner_id);

alter table meta_pending_connections enable row level security;
