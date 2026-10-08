create extension if not exists "pgcrypto";

create table if not exists public.profiles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null unique references auth.users(id) on delete cascade,
  name text,
  avatar_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.products (
  id uuid primary key default gen_random_uuid(),
  asin text not null unique,
  url text not null,
  title text,
  category text,
  brand text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_scraped_at timestamptz
);

create table if not exists public.reviews (
  id uuid primary key default gen_random_uuid(),
  product_id uuid not null references public.products(id) on delete cascade,
  source text not null default 'amazon',
  external_review_id text not null,
  rating integer check (rating between 1 and 5),
  title text,
  content text not null,
  review_date timestamptz,
  verified boolean,
  helpful_votes integer,
  variant text,
  variant_asin text,
  review_country text,
  review_url text,
  images jsonb,
  videos jsonb,
  customers_say text,
  review_aspects jsonb,
  created_at timestamptz not null default now(),
  unique (product_id, source, external_review_id)
);

create table if not exists public.analysis_requests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  product_id uuid not null references public.products(id) on delete cascade,
  product_url text not null,
  product_name text,
  status text not null default 'pending' check (status in ('pending', 'running', 'completed', 'failed')),
  selected_features jsonb,
  feature_scores jsonb,
  result jsonb,
  revify_rating numeric(5, 2),
  error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz
);

create index if not exists reviews_product_id_idx on public.reviews(product_id);
create index if not exists analysis_requests_user_id_idx on public.analysis_requests(user_id);
create index if not exists analysis_requests_product_id_idx on public.analysis_requests(product_id);
create index if not exists analysis_requests_user_created_idx
  on public.analysis_requests(user_id, created_at desc);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

drop trigger if exists products_set_updated_at on public.products;
create trigger products_set_updated_at
before update on public.products
for each row execute function public.set_updated_at();

drop trigger if exists analysis_requests_set_updated_at on public.analysis_requests;
create trigger analysis_requests_set_updated_at
before update on public.analysis_requests
for each row execute function public.set_updated_at();

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (user_id, name, avatar_url)
  values (
    new.id,
    coalesce(
      new.raw_user_meta_data ->> 'name',
      new.raw_user_meta_data ->> 'full_name',
      new.raw_user_meta_data ->> 'user_name'
    ),
    coalesce(
      new.raw_user_meta_data ->> 'avatar_url',
      new.raw_user_meta_data ->> 'picture'
    )
  )
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

alter table public.profiles enable row level security;
alter table public.products enable row level security;
alter table public.reviews enable row level security;
alter table public.analysis_requests enable row level security;

drop policy if exists "Users can read their own profile" on public.profiles;
create policy "Users can read their own profile"
on public.profiles for select to authenticated
using (auth.uid() = user_id);

drop policy if exists "Users can update their own profile" on public.profiles;
create policy "Users can update their own profile"
on public.profiles for update to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "Authenticated users can read products" on public.products;
create policy "Authenticated users can read products"
on public.products for select to authenticated
using (true);

drop policy if exists "Authenticated users can read reviews" on public.reviews;
create policy "Authenticated users can read reviews"
on public.reviews for select to authenticated
using (true);

drop policy if exists "Users can read their own analysis requests" on public.analysis_requests;
create policy "Users can read their own analysis requests"
on public.analysis_requests for select to authenticated
using (auth.uid() = user_id);

drop policy if exists "Users can create their own analysis requests" on public.analysis_requests;
create policy "Users can create their own analysis requests"
on public.analysis_requests for insert to authenticated
with check (auth.uid() = user_id);

drop policy if exists "Users can update their own analysis requests" on public.analysis_requests;
create policy "Users can update their own analysis requests"
on public.analysis_requests for update to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists "Users can delete their own analysis requests" on public.analysis_requests;
create policy "Users can delete their own analysis requests"
on public.analysis_requests for delete to authenticated
using (auth.uid() = user_id);
