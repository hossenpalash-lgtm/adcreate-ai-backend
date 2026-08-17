create table if not exists shopify_connections (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null unique,
  shop_domain text not null,
  access_token text not null,
  connected_at timestamptz not null default now()
);

create index if not exists shopify_connections_owner_id_idx on shopify_connections (owner_id);

alter table shopify_connections enable row level security;
