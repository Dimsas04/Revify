from apify_client import ApifyClient
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revify_flow" / "src"))

from revify_flow.services.analysis_service import (  # noqa: E402
    get_or_create_product,
    upsert_reviews,
)

# Initialize the ApifyClient with your Apify API token
# Replace '<YOUR_API_TOKEN>' with your token.
client = ApifyClient(os.getenv("APIFY_API_TOKEN"))


def _first(item, *keys, default=None):
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def _asin_from_item(item, fallback=None):
    value = _first(
        item, "asin", "ASIN", "productAsin", "product_asin", "productId",
        "product_id", default=fallback
    )
    if not value:
        return None
    match = re.search(r"\b([A-Z0-9]{10})\b", str(value).upper())
    return match.group(1) if match else None


def _json_value(value):
    if value is None or isinstance(value, (dict, list, int, float, bool)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return [value]


def _timestamp(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _boolean(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return str(value).strip().lower() in {"true", "yes", "1", "verified"}


def persist_reviews(items, product_url=None):
    """Persist Zebu dataset items as one product and normalized review rows."""
    items = list(items)
    if not items:
        raise ValueError("Apify returned no review items")

    fallback_asin = _asin_from_item(items[0], product_url)
    if not fallback_asin:
        raise ValueError("Could not determine an ASIN from the Apify results")

    product_url = product_url if str(product_url).startswith("http") else (
        f"https://www.amazon.in/dp/{fallback_asin}"
    )
    scraped_at = datetime.now(timezone.utc).isoformat()
    product = get_or_create_product(
        fallback_asin,
        product_url,
        title=_first(items[0], "productTitle", "product_title", "title", "ProductTitle"),
        brand=_first(items[0], "productBrand", "product_brand", "brand", "Brand"),
        category="Unknown",
        last_scraped_at=scraped_at,
    )

    rows = []
    skipped = 0
    for item in items:
        content = _first(
            item, "ReviewContent", "review_content", "reviewText", "content", "text"
        )
        if not content:
            skipped += 1
            continue

        external_id = _first(
            item, "reviewId", "review_id", "externalReviewId",
            "external_review_id", "id"
        )
        if not external_id:
            identity = "|".join(str(value or "") for value in (
                _asin_from_item(item, fallback_asin),
                _first(item, "reviewDate", "review_date"),
                content,
            ))
            external_id = f"generated-{hashlib.sha256(identity.encode()).hexdigest()}"

        rows.append({
            "product_id": product["id"],
            "source": "amazon",
            "external_review_id": str(external_id),
            "rating": _integer(_first(item, "reviewScore", "review_score", "rating", "stars")),
            "title": _first(item, "reviewTitle", "review_title", "title", "ReviewTitle"),
            "content": str(content),
            "review_date": _timestamp(_first(item, "reviewDate", "review_date", "date", "ReviewDate")),
            "verified": _boolean(_first(item, "verified", "verifiedPurchase", "verified_purchase", "Verified")),
            "helpful_votes": _integer(_first(item, "helpfulVotes", "helpful_votes", "HelpfulVotes")),
            "variant": _first(item, "variant", "Variant"),
            "variant_asin": _first(item, "variantAsin", "variant_asin", "VariantASIN"),
            "review_country": _first(item, "reviewCountry", "review_country", "country", "ReviewCountry"),
            "review_url": _first(item, "reviewUrl", "review_url", "url", "ReviewUrl"),
            "images": _json_value(_first(item, "images", "reviewImages", "review_images")),
            "videos": _json_value(_first(item, "videos", "reviewVideos", "review_videos")),
            "customers_say": _first(item, "CustomersSay", "customersSay", "customers_say", "CustomersSay"),
            "review_aspects": _json_value(
                _first(item, "ReviewAspects", "reviewAspects", "review_aspects", "ReviewAspects")
            ),
        })

    persisted = upsert_reviews(rows)
    print(
        f"Supabase persistence complete: {len(persisted)} reviews upserted, "
        f"{skipped} items skipped without review content, product {product['id']}"
    )
    return {"product": product, "reviews": persisted, "skipped": skipped}

# Input 2: Controlled Scraping using delicious_zebu/amazon-reviews-scraper-with-advanced-filters actor

run_input = {
    "ASIN_or_URL": ["B0BKQRF51Z"],
    "country": "India",

    "unique_only": True,
    "get_customers_say": True,

    "max_reviews": 2,

    "sort_reviews_by": [
        "recent",
        "helpful"
    ],

    "filter_by_verified_purchase_only": [
        "all_reviews"
    ],

    "filter_by_ratings": [
        "all_stars"
    ],

    "filter_by_mediaType": [
        "all_contents"
    ],

    "variant_scope": "all_variants",

    "include_personal_data": False
}

# Run the Actor and wait for it to finish
run = client.actor("delicious_zebu/amazon-reviews-scraper-with-advanced-filters").call(run_input=run_input)

# Fetch and print Actor results from the run's dataset (if there are any)
print(f"💾 Check your data here: https://console.apify.com/storage/datasets/{run.default_dataset_id}")
items = list(client.dataset(run.default_dataset_id).iterate_items())
for item in items:
    print(item)

persist_reviews(items, product_url=run_input["ASIN_or_URL"][0])

# 📚 Want to learn more 📖? Go to → https://docs.apify.com/api/client/python/docs/quick-start