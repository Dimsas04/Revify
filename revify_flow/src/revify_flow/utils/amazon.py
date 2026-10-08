import re


def extract_asin(product_url: str) -> str | None:
    match = re.search(
        r"(?:/dp/|/gp/product/|/product-reviews/)([A-Z0-9]{10})",
        product_url,
        re.IGNORECASE,
    )

    return match.group(1).upper() if match else None