#!/usr/bin/env python3
"""
Amazon Reviews Scraper (with undetected_chromedriver)
----------------------------------------------------
This version:
- Uses undetected_chromedriver (avoids Amazon bot detection)
- Handles login freezes by adding timeouts and JS navigation fallback
- Detects CAPTCHA/2FA and prompts for manual solve
- Keeps same scrape_reviews() and main() structure
"""

import os
import time
import random
import warnings
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Suppress the ChromeDriver cleanup warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
import sys
if sys.platform == 'win32':
    # Suppress Windows handle errors during cleanup
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

# Load credentials
load_dotenv()
USERNAME = os.getenv("NUMBER")
PASSWORD = os.getenv("PASSWORD")

# ---------------------------- Utilities ---------------------------- #

def random_sleep(a=0.4, b=1.0):
    """Short randomized delay to mimic human-like timing."""
    time.sleep(random.uniform(a, b))


def setup_driver():
    """Set up undetected ChromeDriver with minimal automation fingerprints."""
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_experimental_option("prefs", {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
    })

    # Optional: persistent session (keeps you logged in between runs)
    # options.user_data_dir = os.path.join(os.getcwd(), "chrome_profile")

    driver = uc.Chrome(options=options)
    driver.implicitly_wait(2)
    driver.set_page_load_timeout(20)
    return driver


# ---------------------------- Scraper Core ---------------------------- #

def scrape_reviews(url):
    driver = setup_driver()
    data = []

    try:
        driver.get(url)

        # Open the reviews page
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
        ).click()

        # ---------------- LOGIN SECTION ---------------- #
        try:
            try:
                email_input = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "ap_email"))
                )
            except:
                email_input = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "ap_email_login"))
                )

            print(f"Logging in as: {USERNAME}")
            email_input.send_keys(USERNAME)
            random_sleep()

            continue_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.ID, "continue"))
            )
            continue_button.click()

            password_input = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "ap_password"))
            )
            password_input.send_keys(PASSWORD)
            random_sleep()

            sign_in_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.ID, "signInSubmit"))
            )
            sign_in_button.click()

            # Wait for login to complete or timeout safely
            login_ok = False
            try:
                WebDriverWait(driver, 10).until(
                    EC.any_of(
                        EC.url_contains("/youraccount"),
                        EC.presence_of_element_located((By.ID, "nav-link-accountList"))
                    )
                )
                login_ok = True
            except TimeoutException:
                login_ok = False

            # Detect possible CAPTCHA / 2FA page
            page_text = driver.page_source.lower()
            if any(keyword in page_text for keyword in ["captcha", "type the characters", "auth-mfa"]):
                print("⚠️ Detected CAPTCHA or 2FA challenge.")
                input("Please complete it manually in the browser, then press Enter to continue...")

            elif not login_ok:
                print("⚠️ Login not confirmed, continuing anyway...")

        except Exception as e:
            print("⚠️ Login flow skipped or failed:", e)

        # ---------------- POST-LOGIN NAVIGATION ---------------- #
        print("Navigating to target URL...")
        try:
            try:
                driver.get(url)
            except TimeoutException:
                print("Timeout on driver.get(); using JS fallback navigation.")
                driver.execute_script("window.location.href = arguments[0];", url)
                time.sleep(2)
        except Exception as e:
            print("Navigation error:", e)
            driver.execute_script("window.location.href = arguments[0];", url)
            time.sleep(2)

        # Click reviews again after login
        try:
            WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
            ).click()
        except Exception as e:
            print("⚠️ Couldn't click reviews link after login:", e)

        # ---------------- SCRAPING SECTION ---------------- #
        review_titles = []
        reviews = []
        ratings = []
        pages_to_scrape = 10
        for _ in range(pages_to_scrape):
            random_sleep(1, 2)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            with open("amazon_reviews_page.txt", "w", encoding="utf-8") as f:
                f.write(driver.page_source)

            titles = soup.find_all("a", {"data-hook": "review-title"})
            bodies = soup.find_all("span", {"data-hook": "review-body"})
            stars = soup.find_all("span", class_="a-icon-alt")

            for i in range(min(len(titles), len(bodies))):
                review_titles.append(titles[i].text.strip())
                reviews.append(bodies[i].text.strip())
                ratings.append(stars[i].text.strip() if i < len(stars) else "")

            # Go to next page if exists
            try:
                next_btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        '//li[@class="a-last"]/a'
                    ))
                )
                random_sleep(0.5, 1.0)
                next_btn.click()
            except Exception:
                print("No more pages found or next button not clickable.")
                break

        print("✅ Scraping completed successfully!")
        return review_titles, reviews, ratings

    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        return [], [], []
    
    finally:
        # Properly close the driver to avoid handle errors
        try:
            if driver:
                # Close browser windows first
                driver.close()
                # Stop the ChromeDriver service
                if hasattr(driver, 'service') and driver.service:
                    driver.service.stop()
                # Finally quit
                driver.quit()
        except (OSError, Exception) as e:
            # Suppress the WinError 6 handle error - it's harmless
            if "WinError 6" not in str(e) and "handle is invalid" not in str(e):
                print(f"⚠️ Error closing driver: {e}")
            pass


# ---------------------------- Main ---------------------------- #

def main():
    url = "https://www.amazon.in/ASIAN-Wonder-13-Sports-Running-Shoes/dp/B01N54ZM9W/ref=sr_1_5"
    titles, texts, stars = scrape_reviews(url)

    for i in range(len(titles)):
        print(f"{i+1}. {titles[i]} - {stars[i]}")
        print(texts[i])
        print("-" * 80)


if __name__ == "__main__":
    main()
