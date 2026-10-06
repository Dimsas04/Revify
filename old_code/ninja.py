#!/usr/bin/env python3
"""
amazon_reviews_scraper.py

Rewritten to use undetected_chromedriver and a few anti-detection techniques:
- uses undetected_chromedriver (pip install undetected-chromedriver)
- simulates human-like typing and small random delays
- clicks the sign-in button instead of sending ENTER
- waits for login to finish (or prompts for manual CAPTCHA/2FA)
- keeps overall function structure the same (setup_driver, scrape_reviews, main)
"""

import os
import time
import random
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Selenium / undetected chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()
username = os.getenv("NUMBER")
password = os.getenv("PASSWORD")


def random_sleep(a=0.2, b=0.7):
    """Small randomized sleep to mimic human behaviour."""
    time.sleep(random.uniform(a, b))


def human_type(element, text, min_delay=0.03, max_delay=0.12):
    """Type text into an element with short randomized delays between keystrokes."""
    for ch in text:
        element.send_keys(ch)
        time.sleep(random.uniform(min_delay, max_delay))


def setup_driver():
    """
    Returns an undetected chromedriver instance configured to be less detectable.
    If you want to use a persistent profile, set profile_dir to a valid path.
    """
    options = uc.ChromeOptions()
    # Common, safe options
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("prefs", {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
    })

    # You can set a realistic user-agent if desired (optional)
    # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    #                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    # create driver
    driver = uc.Chrome(options=options)
    # small implicit wait (we rely primarily on explicit waits)
    driver.implicitly_wait(2)
    return driver


def _wait_for_login_finish(driver, timeout=20):
    """
    Wait for login to either succeed or require manual verification.
    Returns True if login appears successful, False otherwise (e.g. CAPTCHA/2FA).
    """
    try:
        WebDriverWait(driver, timeout).until(
            EC.any_of(
                EC.url_contains("/gp/your-account"),  # typical account URL (may vary)
                EC.url_contains("/gp/aw/"),  # sometimes used for account pages
                EC.url_contains("/ap/mfa"),  # 2FA page
                EC.presence_of_element_located((By.ID, "auth-error-message-box")),
            )
        )
    except Exception:
        # timed out waiting for expected conditions
        return False

    # Look for obvious signs of success
    current = driver.current_url.lower()
    if any(x in current for x in ("/gp/your-account", "/orders", "youraccount")):
        return True

    # If on a page that indicates 2FA or CAPTCHA, treat as not-successful (manual action needed)
    if "ap/mfa" in current or "captcha" in current or "auth" in current:
        return False

    # default best-effort: consider it successful
    return True


def scrape_reviews(url):
    driver = setup_driver()
    data = []
    driver.get(url)

    # Open the reviews page link on product page
    WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
    ).click()

    try:

        # --- LOGIN SECTION ---
        # Try to find email input; attempt a couple of common IDs (region/site variations)
        try:
            email_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "ap_email"))
            )
        except Exception:
            # fallback to ap_email_login if present
            email_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "ap_email_login"))
            )

        username = os.getenv("NUMBER")
        password = os.getenv("PASSWORD")
        print("Username:", username)

        # Type username in a human-like way
        human_type(email_input, username)
        random_sleep(0.3, 0.6)

        enter_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "continue"))
        )
        enter_button.click()
        random_sleep(0.5, 1.0)

        password_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "ap_password"))
        )
        # Type password human-like
        human_type(password_input, password)
        random_sleep(0.3, 0.6)

        # Click sign in (more reliable than sending ENTER)
        sign_in_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "signInSubmit"))
        )
        sign_in_button.click()

        # Wait for login to complete or detect CAPTCHA/2FA
        ok = _wait_for_login_finish(driver, timeout=15)
        if not ok:
            print("Login might require manual verification (CAPTCHA or 2FA).")
            print("Please solve it in the opened browser window.")
            input("After completing verification, press Enter to continue...")

    except Exception as e:
        # something in the login flow might have failed; we continue but warn
        print("Login section encountered an issue (continuing anyway):", e)

    # --- POST LOGIN NAVIGATION ---
    # Navigate back to product reviews page and continue scraping
    print("Proceeding to target URL...")
    try:
        driver.get(url)
    except Exception as e:
        print("Navigation failed on first attempt, retrying:", e)
        time.sleep(2)
        driver.get(url)

    # Click reviews link again (product page might vary, but this matches original flow)
    try:
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
        ).click()
    except Exception as e:
        print("Couldn't click reviews link after login:", e)
        # continue; sometimes product pages show reviews differently

    # --- SCRAPING SECTION (unchanged structure) ---
    a = 7
    review_titles = []
    reviews = []
    ratings = []

    while a != 0:
        a -= 1
        # small random delay to avoid rapid-fire requests
        random_sleep(0.8, 1.6)

        tables = driver.page_source
        soup = BeautifulSoup(tables, "html.parser")

        with open("amazon_reviews_page.txt", "w", encoding="utf-8") as f:
            f.write(tables)

        ret_bodies = soup.find_all("span", {"data-hook": "review-body"})
        ret_titles = soup.find_all("a", {"data-hook": "review-title"})
        ret_ratings = soup.find_all("span", class_="a-icon-alt")

        for i in range(0, len(ret_bodies)):
            # defensive checks (some pages may have mismatched lengths)
            try:
                t = ret_titles[i].text.strip()
            except Exception:
                t = ""
            try:
                b = ret_bodies[i].text.strip()
            except Exception:
                b = ""
            try:
                r = ret_ratings[i].text.strip()
            except Exception:
                r = ""

            review_titles.append(t)
            reviews.append(b)
            ratings.append(r)

        # Try to click next page in reviews
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, '/html/body/div[1]/div[2]/div/div[1]/div/div[1]/div[5]/div[3]/div/div/span/div/ul/li[2]/a')
                )
            )
            # small pause and click
            random_sleep(0.3, 0.7)
            next_button.click()
            # give the new page a moment to load
            random_sleep(1.0, 2.0)
        except Exception:
            break

    # (Optionally) keep the browser open so user can inspect, because earlier code set detach behavior
    return review_titles, reviews, ratings


def main():
    url = "https://www.amazon.in/ASIAN-Wonder-13-Sports-Running-Shoes/dp/B01N54ZM9W/ref=sr_1_5"
    reviews_titles, reviews_list, ratings = scrape_reviews(url)

    for i in range(len(reviews_titles)):
        print(reviews_titles[i])
        print(reviews_list[i])
        print("\n\n")


if __name__ == "__main__":
    main()
