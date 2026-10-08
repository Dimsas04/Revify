from datetime import datetime, timezone
from typing import Any

from .supabase_client import get_supabase_client


def create_analysis_request(user_id: str, product_url: str, product_name: str = "",
                            selected_features: list[str] | None = None,
                            product_id: str | None = None) -> str:
    payload: dict[str, Any] = {
        "user_id": user_id,
        "product_url": product_url,
        "product_name": product_name or None,
        "status": "running",
        "selected_features": selected_features,
        "product_id": product_id,
    }
    response = get_supabase_client().table("analysis_requests").insert(payload).execute()
    if not response.data:
        raise RuntimeError("Supabase did not create the analysis request")
    return response.data[0]["id"]


def update_analysis_request(request_id: str, *, status: str, result: dict | None = None,
                            error: str | None = None) -> None:
    payload: dict[str, Any] = {"status": status, "error": error}
    if result is not None:
        payload["result"] = result
    if status == "completed":
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    get_supabase_client().table("analysis_requests").update(payload).eq("id", request_id).execute()


# def get_analysis_request(
#     request_id: str,
#     user_id: str,
# ) -> dict | None:
#     response = (
#         get_supabase_client()
#         .table("analysis_requests")
#         .select("*")
#         .eq("id", request_id)
#         .eq("user_id", user_id)
#         .limit(1)
#         .execute()
#     )

#     return response.data[0] if response.data else None

def get_analysis_request(
    request_id: str,
    user_id: str | None = None,
) -> dict | None:

    query = (
        get_supabase_client()
        .table("analysis_requests")
        .select("*")
        .eq("id", request_id)
    )

    if user_id:
        query = query.eq("user_id", user_id)

    response = (
        query
        .limit(1)
        .execute()
    )

    return response.data[0] if response.data else None


def set_extracted_features(
    request_id: str,
    features: list[str],
) -> None:
    (
        get_supabase_client()
        .table("analysis_requests")
        .update({
            "extracted_features": features,
        })
        .eq("id", request_id)
        .execute()
    )

def update_analysis_request_features(
    request_id: str,
    selected_features: list[str],
) -> None:
    (
        get_supabase_client()
        .table("analysis_requests")
        .update({
            "selected_features": selected_features,
        })
        .eq("id", request_id)
        .execute()
    )

# def update_analysis_progress(
#     request_id: str,
#     progress: int,
#     current_phase: str,
# ) -> None:
#     (
#         get_supabase_client()
#         .table("analysis_requests")
#         .update({
#             "progress": progress,
#             "current_phase": current_phase,
#         })
#         .eq("id", request_id)
#         .execute()
#     )

def update_analysis_progress(
    request_id: str,
    progress: int,
    current_phase: str,
) -> None:

    (
        get_supabase_client()
        .table("analysis_requests")
        .update({
            "progress": progress,
            "current_phase": current_phase,
        })
        .eq("id", request_id)
        .execute()
    )

def list_user_analysis_requests(user_id: str) -> list[dict]:
    response = (
        get_supabase_client().table("analysis_requests")
        .select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    )
    return response.data or []


def get_or_create_product(asin: str, url: str, title: str | None = None,
                          brand: str | None = None,
                          category: str | None = None,
                          last_scraped_at: str | None = None) -> dict:
    payload: dict[str, Any] = {
        "asin": asin,
        "url": url,
        "title": title,
        "brand": brand or "Unknown",
        "category": category or "Unknown",
    }
    if last_scraped_at:
        payload["last_scraped_at"] = last_scraped_at
    response = get_supabase_client().table("products").upsert(
        payload,
        on_conflict="asin",
    ).execute()
    if not response.data:
        raise RuntimeError("Supabase did not create or retrieve the product")
    return response.data[0]


def get_product_by_asin(asin: str) -> dict | None:
    response = (
        get_supabase_client().table("products")
        .select("*").eq("asin", asin).limit(1).execute()
    )
    return response.data[0] if response.data else None


def upsert_reviews(reviews: list[dict]) -> list[dict]:
    if not reviews:
        return []
    response = (
        get_supabase_client().table("reviews")
        .upsert(reviews, on_conflict="product_id,source,external_review_id")
        .execute()
    )
    return response.data or []


def get_product_reviews(product_id: str) -> list[dict]:
    response = (
        get_supabase_client().table("reviews")
        .select("*").eq("product_id", product_id)
        .order("review_date", desc=True).execute()
    )
    return response.data or []


def get_product_reviews_by_asin(asin: str) -> tuple[dict | None, list[dict]]:
    product = get_product_by_asin(asin)
    if not product:
        return None, []
    return product, get_product_reviews(product["id"])
