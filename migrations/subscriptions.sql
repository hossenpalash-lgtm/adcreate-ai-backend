create table if not exists subscriptions (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null unique,
  stripe_customer_id text not null,
  stripe_subscription_id text not null unique,
  price_id text not null,
  tier text not null,
  status text not null,
  current_period_end timestamptz,
  last_invoice_id text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists subscriptions_owner_id_idx on subscriptions (owner_id);

alter table subscriptions enable row level security;
