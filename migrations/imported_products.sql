create table if not exists imported_products (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null,
  name text not null,
  description text,
  price text,
  image_url text,
  created_at timestamptz not null default now()
);

create index if not exists imported_products_owner_id_idx on imported_products (owner_id);

alter table imported_products enable row level security;
