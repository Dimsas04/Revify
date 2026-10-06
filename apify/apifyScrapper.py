from apify_client import ApifyClient
import os

# Initialize the ApifyClient with your Apify API token
# Replace '<YOUR_API_TOKEN>' with your token.
client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

# Prepare the Actor input
# Junglee: https://apify.com/junglee/amazon-reviews-scraper
# run_input = {
#     "productUrls": [{ "url": "https://www.amazon.in/Concept-Kart-Earphone-Detachable-Cyan/dp/B0BKQRF51Z/" }],
#     "maxReviews": 100,
#     "scrapeProductDetails": False,
#     "reviewsAlwaysSaveCategoryData": False,
#     "deduplicateRedirectedAsins": True,
# }

# # Run the Actor and wait for it to finish
# run = client.actor("junglee/amazon-reviews-scraper").call(run_input=run_input)

# # Fetch and print Actor results from the run's dataset (if there are any)
# print(f"💾 Check your data here: https://console.apify.com/storage/datasets/{run.default_dataset_id}")
# for item in client.dataset(run.default_dataset_id).iterate_items():
#     print(item)


# jdtpnjtp/amazon-reviews actor: 
# run_input = {
#     "products": ["B0BKQRF51Z"],
#     "country": "us",
#     "starFilter": "all_stars",
#     "sort": "recent",
#     "maxReviewsPerAsin": 50,
#     "includeTopReviewsFallback": False,
#     "stopOnError": False,
# }

# # Run the Actor and wait for it to finish
# run = client.actor("y2wdujRKD2FUa5h5l").call(run_input=run_input)

# # Fetch and print Actor results from the run's dataset (if there are any)
# # for item in client.dataset(run["defaultDatasetId"]).iterate_items():
# #     print(item)

# print(
#     f"Dataset: https://console.apify.com/storage/datasets/"
#     f"{run.default_dataset_id}"
# )

# items = list(
#     client.dataset(run.default_dataset_id).iterate_items()
# )

# print(f"\nReviews returned: {len(items)}\n")

# for item in items[:5]:
#     print(item)

# Result: Limited to 10 reviws per ASIN, so only 10 reviews were returned for the product.

# themineworks/amazon-reviews actor: 
# run_input = {
#     "asins": ["B08N5KWB9H"],
#     "marketplace": None,
#     "maxReviews": 13,
#     "sortBy": None,
#     "filterByStar": None,
#     "includeInternational": None,
#     "residentialFallback": None,
#     "proxyConfig": None,
# }

# # Run the Actor and wait for it to finish
# run = client.actor("themineworks/amazon-reviews").call(run_input=run_input)

# # Fetch and print Actor results from the run's dataset (if there are any)
# print(f"💾 Check your data here: https://console.apify.com/storage/datasets/{run.default_dataset_id}")
# for item in client.dataset(run.default_dataset_id).iterate_items():
#     print(item)

# Result: Did not have IN marketplace, so used US marketplace instead, where the product was not found. So, the result was empty.

# hypnotic_freedom/amazon-reviews-scraper actor:
# run_input = {
#     "asin": "B0BKQRF51Z",
#     "marketplace": "amazon.de"
# }

# # Run the Actor and wait for it to finish
# run = client.actor("hypnotic_freedom/amazon-reviews-scraper").call(run_input=run_input)

# # Fetch and print Actor results from the run's dataset (if there are any)
# print(f"💾 Check your data here: https://console.apify.com/storage/datasets/{run.default_dataset_id}")
# for item in client.dataset(run.default_dataset_id).iterate_items():
#     print(item)

# 📚 Want to learn more 📖? Go to → https://docs.apify.com/api/client/python/docs/quick-start

# Result: Not available for India


# delicious_zebu/amazon-reviews-scraper-with-advanced-filters actor:

# Prepare the Actor input

# Input 1: Aggressive scraping, all reviews, all variants, all filters applied
# run_input = {
#     "ASIN_or_URL": ["B0BKQRF51Z"],
#     "country": "India",

#     "recent_days": 0,

#     "unique_only": True,

#     "get_customers_say": True,

#     "max_reviews": 100,

#     "sort_reviews_by": [
#         "helpful",
#         "recent"
#     ],

#     "filter_by_verified_purchase_only": [
#         "all_reviews",
#         "avp_only_reviews"
#     ],

#     "filter_by_ratings": [
#         "all_stars",
#         "five_star",
#         "four_star",
#         "three_star",
#         "two_star",
#         "one_star",
#         "positive",
#         "critical"
#     ],

#     "filter_by_mediaType": [
#         "all_contents",
#         "media_reviews_only"
#     ],

#     "variant_scope": "all_variants",

#     "include_personal_data": False
# }


# Input 2: Controlled Scraping using delicious_zebu/amazon-reviews-scraper-with-advanced-filters actor

run_input = {
    "ASIN_or_URL": ["B0BKQRF51Z"],
    "country": "India",

    "unique_only": True,
    "get_customers_say": True,

    "max_reviews": 200,

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
for item in client.dataset(run.default_dataset_id).iterate_items():
    print(item)

# 📚 Want to learn more 📖? Go to → https://docs.apify.com/api/client/python/docs/quick-start