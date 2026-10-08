"""Reusable Apify review acquisition and Supabase persistence tool."""

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any

from apify_client import ApifyClient
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..services.analysis_service import (
    get_or_create_product,
    upsert_reviews,
)

_tool_path = os.path.abspath(__file__)
_backend_env_path = os.path.abspath(
    os.path.join(os.path.dirname(_tool_path), "..", "..", "..", ".env")
)
_repository_env_path = os.path.abspath(
    os.path.join(os.path.dirname(_tool_path), "..", "..", "..", "..", ".env")
)

# The backend env contains Supabase settings; the repository env contains the
# existing Apify token. Keep both explicit because load_dotenv() uses cwd.
load_dotenv(_backend_env_path, override=False)
load_dotenv(_repository_env_path, override=False)


class ApifyScraperSchema(BaseModel):
    product_url: str = Field(description="Amazon product URL or ASIN")
    max_reviews: int = Field(default=200, description="Maximum reviews to collect")


def _first(item: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def _asin(value: Any) -> str | None:
    match = re.search(r"\b([A-Z0-9]{10})\b", str(value or "").upper())
    return match.group(1) if match else None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return [value]


def _timestamp(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).strip().lower() in {"true", "yes", "1", "verified"}


def persist_apify_reviews(items: list[dict[str, Any]], product_url: str) -> dict[str, Any]:
    if not items:
        raise ValueError("Apify returned no review items")

    asin = next((_asin(_first(item, "asin", "ASIN", "productAsin", "productId"))
                 for item in items), None) or _asin(product_url)
    if not asin:
        raise ValueError("Could not determine an ASIN from the Apify result or URL")

    product = get_or_create_product(
        asin,
        product_url if product_url.startswith("http") else f"https://www.amazon.in/dp/{asin}",
        title=_first(items[0], "productTitle", "product_title"),
        brand="Unknown",
        category="Unknown",
        last_scraped_at=datetime.now(timezone.utc).isoformat(),
    )
    rows = []
    for item in items:
        content = _first(item, "reviewContent", "review_content", "reviewText", "content", "text")
        if not content:
            continue
        external_id = _first(item, "reviewId", "review_id", "externalReviewId", "id")
        if not external_id:
            identity = "|".join((
                asin,
                str(_first(item, "reviewDate", "review_date", "date") or ""),
                str(content),
            ))
            external_id = f"generated-{hashlib.sha256(identity.encode()).hexdigest()}"
        rows.append({
            "product_id": product["id"],
            "source": "amazon",
            "external_review_id": str(external_id),
            "rating": _integer(_first(item, "reviewScore", "review_score", "rating", "stars")),
            "title": _first(item, "reviewTitle", "review_title", "title"),
            "content": str(content),
            "review_date": _timestamp(_first(item, "reviewDate", "review_date", "date")),
            "verified": _boolean(_first(item, "verified", "verifiedPurchase", "verified_purchase")),
            "helpful_votes": _integer(_first(item, "helpfulVotes", "helpful_votes")),
            "variant": _first(item, "variant"),
            "variant_asin": _first(item, "variantAsin", "variant_asin"),
            "review_country": _first(item, "reviewCountry", "review_country", "country"),
            "review_url": _first(item, "reviewUrl", "review_url", "url"),
            "images": _json_value(_first(item, "images", "reviewImages", "review_images")),
            "videos": _json_value(_first(item, "videos", "reviewVideos", "review_videos")),
            "customers_say": _first(item, "CustomersSay", "customersSay", "customers_say"),
            "review_aspects": _json_value(
                _first(item, "ReviewAspects", "reviewAspects", "review_aspects")
            ),
        })
    return {"product": product, "reviews": upsert_reviews(rows)}


def scrape_and_persist_reviews(product_url: str, max_reviews: int = 10) -> dict[str, Any]:
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise RuntimeError("APIFY_API_TOKEN must be configured")
    asin_or_url = product_url
    run_input = {
        "ASIN_or_URL": [asin_or_url],
        "country": "India",
        "unique_only": True,
        "get_customers_say": True,
        "max_reviews": max_reviews,
        "sort_reviews_by": ["recent", "helpful"],
        "filter_by_verified_purchase_only": ["all_reviews"],
        "filter_by_ratings": ["all_stars"],
        "filter_by_mediaType": ["all_contents"],
        "variant_scope": "all_variants",
        "include_personal_data": False,
    }
    client = ApifyClient(token)
    run = client.actor(
        "delicious_zebu/amazon-reviews-scraper-with-advanced-filters"
    ).call(run_input=run_input)
    items = list(client.dataset(run.default_dataset_id).iterate_items())
    return persist_apify_reviews(items, product_url)


class ApifyScraperTool(BaseTool):
    name: str = "apify_amazon_reviews_tool"
    description: str = "Collects Amazon reviews with Apify and persists them in Supabase."
    args_schema: type[BaseModel] = ApifyScraperSchema

    def _run(self, product_url: str, max_reviews: int = 10) -> str:
        result = scrape_and_persist_reviews(product_url, max_reviews)
        return f"Persisted {len(result['reviews'])} reviews for product {result['product']['id']}"
