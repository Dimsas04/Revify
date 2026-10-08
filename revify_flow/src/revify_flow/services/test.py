import os

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".." , ".env")
load_dotenv(env_path)

def test_supabase_configuration():
    print("Testing Supabase configuration...")
    url = os.getenv("SUPABASE_URL")
    print(f"SUPABASE_URL: {url}")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")
    print(f"SUPABASE_SECRET_KEY: {secret_key}")
    if not url or not secret_key:
        print("SUPABASE_URL and SUPABASE_SECRET_KEY must be configured")

test_supabase_configuration()