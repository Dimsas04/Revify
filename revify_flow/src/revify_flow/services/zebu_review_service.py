import os
from datetime import datetime, timezone
from dateutil import parser

from apify_client import ApifyClient

from .analysis_service import (
    get_or_create_product,
    get_product_reviews_by_asin,
    upsert_reviews,
)


ZEBU_ACTOR_ID = "delicious_zebu/amazon-reviews-scraper-with-advanced-filters"


def _first(item: dict, *keys: str, default=None):
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def _default_dataset_id(run) -> str:
    if isinstance(run, dict):
        dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
    else:
        dataset_id = getattr(run, "default_dataset_id", None)
        if not dataset_id:
            dataset_id = getattr(run, "defaultDatasetId", None)

    if not dataset_id:
        raise RuntimeError(
            "Apify run did not return a default dataset ID"
        )

    return str(dataset_id)


def _parse_integer(value, *, minimum=None, maximum=None):
    if value is None or value == "":
        return None

    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None

    if minimum is not None and parsed < minimum:
        return None
    if maximum is not None and parsed > maximum:
        return None
    return parsed


def _parse_date(value):
    if not value:
        return None

    try:
        return parser.parse(str(value)).date().isoformat()
    except Exception:
        return None


def _normalize_review(item: dict, product_id: str) -> dict | None:
    review_id = _first(item, "ReviewId", "reviewId", "review_id", "externalReviewId", "id")
    content = _first(
        item,
        "ReviewContent",
        "reviewContent",
        "review_content",
        "reviewText",
        "content",
        "text",
    )

    if not review_id or not content:
        return None

    return {
        "product_id": product_id,
        "source": "amazon",
        "external_review_id": str(review_id),
        "rating": _parse_integer(
            _first(item, "ReviewScore", "reviewScore", "review_score", "rating", "stars"),
            minimum=1,
            maximum=5,
        ),
        "title": _first(item, "ReviewTitle", "reviewTitle", "review_title", "title"),
        "content": content,
        "review_date": _parse_date(
            _first(item, "ReviewDate", "reviewDate", "review_date", "date")
        ),
        "verified": bool(
            _first(item, "Verified", "verified", "verifiedPurchase", "verified_purchase")
        ),
        "helpful_votes": _parse_integer(
            _first(item, "HelpfulVotes", "helpfulVotes", "helpful_votes")
        ) or 0,
        "variant": _first(item, "Variant", "variant"),
        "variant_asin": _first(item, "VariantASIN", "variantAsin", "variant_asin"),
        "review_country": _first(item, "ReviewCountry", "reviewCountry", "review_country", "country"),
        "review_url": _first(item, "ReviewUrl", "reviewUrl", "review_url", "url"),
        "images": _first(item, "Images", "images", "reviewImages", "review_images"),
        "videos": _first(item, "Videos", "videos", "reviewVideos", "review_videos"),
        "customers_say": _first(item, "CustomersSay", "customersSay", "customers_say"),
        "review_aspects": _first(item, "ReviewAspects", "reviewAspects", "review_aspects"),
    }


def fetch_and_persist_reviews(
    asin: str,
    product_url: str,
    product_name: str | None = None,
) -> list[dict]:
    existing_product, existing_reviews = get_product_reviews_by_asin(asin)
    if existing_product and existing_reviews:
        print(
            f"Reusing {len(existing_reviews)} existing reviews for ASIN {asin}; "
            "skipping Apify"
        )
        return existing_reviews

    token = os.getenv("APIFY_API_TOKEN")

    if not token:
        raise RuntimeError("APIFY_API_TOKEN is not configured")

    client = ApifyClient(token)

    run_input = {
        "ASIN_or_URL": [asin],
        "country": "India",
        "End_date": "1990-01-01",
        "recent_days": 0,
        "unique_only": True,
        "get_customers_say": True,
        "max_reviews": 200,
        "sort_reviews_by": [
            "helpful",
            "recent",
        ],
        "filter_by_verified_purchase_only": [
            "all_reviews",
            "avp_only_reviews",
        ],
        "filter_by_ratings": [
            "all_stars",
            "five_star",
            "four_star",
            "three_star",
            "two_star",
            "one_star",
            "positive",
            "critical",
        ],
        "filter_by_mediaType": [
            "all_contents",
            "media_reviews_only",
        ],
        "variant_scope": "all_variants",
        "filter_by_keywords": [],
        "include_personal_data": False,
    }

    run = client.actor(ZEBU_ACTOR_ID).call(
        run_input=run_input
    )

    dataset_id = _default_dataset_id(run)
    items = list(client.dataset(dataset_id).iterate_items())

    if not items:
        raise RuntimeError("Zebu returned no reviews")

    first = items[0]

    product = get_or_create_product(
        asin=asin,
        url=product_url,
        title=_first(first, "ProductTitle", "productTitle", "product_title") or product_name,
        brand=_first(first, "Brand", "brand"),
    )

    normalized_reviews = []

    for item in items:
        normalized = _normalize_review(
            item,
            product["id"],
        )

        if normalized:
            normalized_reviews.append(normalized)

    if not normalized_reviews:
        raise RuntimeError(
            "Zebu returned reviews, but none contained usable review content"
        )

    persisted_reviews = upsert_reviews(normalized_reviews)

    get_or_create_product(
        asin=asin,
        url=product_url,
        title=_first(first, "ProductTitle", "productTitle", "product_title") or product_name,
        brand=_first(first, "Brand", "brand"),
        last_scraped_at=datetime.now(timezone.utc).isoformat(),
    )

    return persisted_reviews