create table if not exists meta_connections (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null unique,
  page_id text not null,
  page_name text not null,
  page_access_token text not null,
  ig_user_id text,
  ig_username text,
  connected_at timestamptz not null default now()
);

create index if not exists meta_connections_owner_id_idx on meta_connections (owner_id);

alter table meta_connections enable row level security;
