import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not secret_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be configured")
    return create_client(url, secret_key)


def get_authenticated_user(access_token: str):
    if not access_token:
        return None
    try:
        response = get_supabase_client().auth.get_user(access_token)
        return response.user
    except Exception:
        return None


def require_supabase_configuration() -> None:
    get_supabase_client()
