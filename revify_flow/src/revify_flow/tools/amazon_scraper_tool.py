# from typing import Optional
# from crewai.tools import tool
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# import time
# import os
# import sys
# import pandas as pd
# from dotenv import load_dotenv

# @tool()
# class AmazonScraperTool():
#     name: str = "amazon_scraper_tool"
#     description: str = "Scrapes customer reviews from Amazon product pages"

#     def __init__(self):
#         super().__init__(
#             name=self.name,
#             description=self.description,
#             func=self._scrape_reviews
#         )
        
#         # Load environment variables
#         load_dotenv()
#         self.username = os.getenv('NUMBER')
#         self.password = os.getenv('PASSWORD')
        
#         # Check if credentials are available
#         if not self.username or not self.password:
#             print("⚠️ Error: Missing credentials in .env file")
#             print("Please ensure your .env file contains:")
#             print("NUMBER=your_email_or_phone_number")
#             print("PASSWORD=your_amazon_password")

#     def __call__(self, url: str) -> str:
#         # wrap your existing `_scrape_reviews()` here
#         return self._scrape_reviews(url)

#     def _scrape_reviews(self, url: str, target_reviews: int = 50, product_name: Optional[str] = None) -> str:
#         """
#         Scrape reviews from Amazon product page
        
#         Args:
#             url: Amazon product URL
#             target_reviews: Number of reviews to scrape (default: 50)
#             product_name: Optional name to use for the product (default: extracted from page)
            
#         Returns:
#             Path to the CSV file containing the scraped reviews
#         """
#         # Check for credentials
#         if not self.username or not self.password:
#             return "Error: Missing Amazon credentials in .env file"
        
#         print(f"Starting to scrape {target_reviews} reviews from: {url}")
        
#         # Set up the driver and scrape reviews
#         try:
#             reviews_titles, reviews_texts, ratings = self._perform_scraping(url, target_reviews)
            
#             # Save to CSV
#             if not product_name:
#                 product_name = "Amazon Product"  # Default name if not provided
                
#             csv_path = self._save_to_csv(reviews_titles, reviews_texts, ratings, product_name)
            
#             return f"Successfully scraped {len(reviews_texts)} reviews. Data saved to {csv_path}"
        
#         except Exception as e:
#             return f"Error during scraping: {str(e)}"

#     def _perform_scraping(self, url, target_reviews):
#         """Internal method to handle the actual scraping logic"""
#         driver = self._setup_driver()
#         review_titles = []
#         reviews = []
#         ratings = []

#         try:
#             print(f"Opening URL: {url}")
#             driver.get(url)

#             # Wait and click on the "See all reviews" link
#             print("Looking for 'See all reviews' link...")
#             WebDriverWait(driver, 20).until(
#                 EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
#             ).click()
            
#             # Wait for email input field and enter email
#             print("Logging in with provided credentials...")
#             email_input = WebDriverWait(driver, 20).until(
#                 EC.presence_of_element_located((By.ID, "ap_email_login"))
#             )
#             print(f"Using username: {self.username[:2]}...{self.username[-2:]}")
#             email_input.send_keys(self.username)
            
#             enter_button = WebDriverWait(driver, 20).until(
#                 EC.element_to_be_clickable((By.ID, "continue"))
#             )
#             enter_button.click()
            
#             password_input = WebDriverWait(driver, 20).until(
#                 EC.presence_of_element_located((By.ID, "ap_password"))
#             )
#             password_input.send_keys(self.password)
            
#             sign_in_button = WebDriverWait(driver, 20).until(
#                 EC.element_to_be_clickable((By.ID, "signInSubmit"))
#             )
#             sign_in_button.click()
            
#             page_count = 1
#             max_pages = 10  # Maximum number of pages to scrape
            
#             # Continue until we reach target_reviews or max_pages
#             while len(reviews) < target_reviews and page_count <= max_pages:
#                 print(f"Scraping page {page_count}...")
#                 # Wait for reviews to load
#                 time.sleep(5)  # Give more time for page to load
                
#                 tables = driver.page_source
                
#                 soup = BeautifulSoup(tables, 'html.parser')
#                 ret_bodies = soup.find_all('span', {'data-hook': 'review-body'})
#                 ret_titles = soup.find_all('a', {'data-hook': 'review-title'})  
#                 ret_ratings = soup.find_all('span', class_='a-icon-alt')
                
#                 print(f"Found {len(ret_bodies)} reviews on this page")
                
#                 # Process found reviews up to target_reviews
#                 for i in range(min(len(ret_bodies), target_reviews - len(reviews))):
#                     review_titles.append(ret_titles[i].text.strip())
#                     reviews.append(ret_bodies[i].text.strip())
#                     ratings.append(ret_ratings[i].text.strip())
                
#                 # Stop if we've reached our target
#                 if len(reviews) >= target_reviews:
#                     print(f"✅ Reached target of {target_reviews} reviews.")
#                     break
                    
#                 # Try to go to next page
#                 try:
#                     print("Looking for next page button...")
#                     next_button = WebDriverWait(driver, 10).until(
#                         EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
#                     )
#                     next_button.click()
#                     page_count += 1
#                     time.sleep(2)  # Brief pause after clicking
#                 except Exception as e:
#                     print(f"No more pages available: {str(e)}")
#                     break
                    
#             print(f"Scraping complete. Retrieved {len(reviews)} reviews.")
            
#         except Exception as e:
#             print(f"Error during scraping: {str(e)}")
#         finally:
#             driver.quit()
        
#         # Limit to target_reviews in case we got more
#         return review_titles[:target_reviews], reviews[:target_reviews], ratings[:target_reviews]

#     def _setup_driver(self):
#         """Set up the Selenium WebDriver with appropriate options"""
#         chrome_options = Options()
#         chrome_options.add_argument('--disable-gpu')
#         chrome_options.add_argument('--disable-dev-shm-usage')
#         chrome_options.add_argument('--no-sandbox')
#         chrome_options.add_argument('--log-level=3')  # Minimize logging
#         chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
#         chrome_options.add_experimental_option('useAutomationExtension', False)
#         chrome_options.add_experimental_option("detach", True)  # Keep browser open after script finishes
#         prefs = {
#             "credentials_enable_service": False,         # Disables the password manager
#             "profile.password_manager_enabled": False    # Disables the password manager UI
#         }
#         chrome_options.add_experimental_option("prefs", prefs)
        
#         return webdriver.Chrome(options=chrome_options)

#     def _save_to_csv(self, titles, reviews, ratings, product_name="Amazon Product"):
#         """Save reviews to CSV in format compatible with preprocessing.ipynb"""
#         # Create a DataFrame
#         df = pd.DataFrame({
#             "id": range(1, len(titles) + 1),
#             "name": product_name,
#             "brand": "Amazon",
#             "categories": "Product",
#             "primaryCategories": "Product",
#             "reviews.doRecommend": [True if r.startswith("5") or r.startswith("4") else False for r in ratings],
#             "reviews.rating": [float(r.split()[0]) if r and r[0].isdigit() else 0 for r in ratings],
#             "reviews.text": reviews,
#             "reviews.title": titles
#         })
        
#         # Save to CSV
#         filename = "scraped_reviews.csv"
#         df.to_csv(filename, index=False)
#         print(f"✅ Saved {len(df)} reviews to {filename}")
#         return filename

# Version 2: 

# from typing import ClassVar, Optional, Dict, Any, Type
# from pydantic import BaseModel, Field, PrivateAttr
# from crewai.tools import BaseTool
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# import time
# import os
# import pandas as pd
# from dotenv import load_dotenv

# # Define the input schema using Pydantic
# class AmazonScraperSchema(BaseModel):
#     product_url: str = Field(
#         description="Amazon product URL to scrape reviews from"
#     )
#     target_reviews: int = Field(
#         description="Number of reviews to scrape"
#     )
#     product_name: Optional[str] = Field(
#         default=None,
#         description="Optional name to use for the product (default: extracted from page)"
#     )

# class AmazonScraperTool(BaseTool):
#     """Tool that scrapes customer reviews from Amazon product pages"""

#     name: str = "amazon_scraper_tool"
#     description: str = "Scrapes customer reviews from Amazon product pages"
#     args_schema: Type[BaseModel] = AmazonScraperSchema

#     # Use class variables to store credentials
#     _username: str = PrivateAttr()
#     _password: str = PrivateAttr()

#     def __init__(self):
#         # First initialize the parent class
#         super().__init__()

#         # Load environment variables
#         load_dotenv()
#         self._username = os.getenv('NUMBER')
#         self._password = os.getenv('PASSWORD')

#         # Check if credentials are available
#         if not self._username or not self._password:
#             print("⚠️ Error: Missing credentials in .env file")
#             print("Please ensure your .env file contains:")
#             print("NUMBER=your_email_or_phone_number")
#             print("PASSWORD=your_amazon_password")
#         else:
#             print(f"✓ Credentials loaded successfully (username: {self._username[:2]}...{self._username[-2:]})")
    
#     def _run(self, product_url: str, target_reviews: int = 50, product_name: Optional[str] = None) -> str:
        
#         """Run the tool with the given inputs."""
#         print(f"DEBUG: Received URL to scrape: {product_url}")
#         # Check for credentials
#         if not self._username or not self._password:
#             return "Error: Missing Amazon credentials in .env file"
#         # Check for credentials
#         # if not AmazonScraperTool._username or not AmazonScraperTool._password:
#         #     return "Error: Missing Amazon credentials in .env file"
        
#         print(f"Starting to scrape {target_reviews} reviews from: {product_url}")
        
#         # Set up the driver and scrape reviews
#         try:
#             reviews_titles, reviews_texts, ratings = self._perform_scraping(product_url, target_reviews)
            
#             # Save to CSV
#             if not product_name:
#                 product_name = "Amazon Product"  # Default name if not provided
                
#             csv_path = self._save_to_csv(reviews_titles, reviews_texts, ratings, product_name)
            
#             return f"Successfully scraped {len(reviews_texts)} reviews. Data saved to {csv_path}"
        
#         except Exception as e:
#             return f"Error during scraping: {str(e)}"

#     def _perform_scraping(self, product_url : str, target_reviews):
#         """Internal method to handle the actual scraping logic"""
#         driver = self._setup_driver()
#         review_titles = []
#         reviews = []
#         ratings = []

#         try:
#             print(f"Opening URL: {product_url}")
#             driver.get(product_url)

#             # Wait and click on the "See all reviews" link
#             print("Looking for 'See all reviews' link...")
#             WebDriverWait(driver, 20).until(
#                 EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
#             ).click()
            
#             # Wait for email input field and enter email
#             print("Logging in with provided credentials...")
#             email_input = WebDriverWait(driver, 20).until(
#                 EC.presence_of_element_located((By.ID, "ap_email_login"))
#             )
#             print(f"Using username: {self._username[:2]}...{self._username[-2:]}")
#             email_input.send_keys(self._username)
            
#             enter_button = WebDriverWait(driver, 20).until(
#                 EC.element_to_be_clickable((By.ID, "continue"))
#             )
#             enter_button.click()
            
#             password_input = WebDriverWait(driver, 20).until(
#                 EC.presence_of_element_located((By.ID, "ap_password"))
#             )
#             password_input.send_keys(self._password)
            
#             sign_in_button = WebDriverWait(driver, 20).until(
#                 EC.element_to_be_clickable((By.ID, "signInSubmit"))
#             )
#             sign_in_button.click()
            
#             page_count = 1
#             max_pages = 10  # Maximum number of pages to scrape
            
#             # Continue until we reach target_reviews or max_pages
#             while len(reviews) < target_reviews and page_count <= max_pages:
#                 print(f"Scraping page {page_count}...")
#                 # Wait for reviews to load
#                 time.sleep(5)  # Give more time for page to load
                
#                 tables = driver.page_source
                
#                 soup = BeautifulSoup(tables, 'html.parser')
#                 ret_bodies = soup.find_all('span', {'data-hook': 'review-body'})
#                 ret_titles = soup.find_all('a', {'data-hook': 'review-title'})  
#                 ret_ratings = soup.find_all('span', class_='a-icon-alt')
                
#                 print(f"Found {len(ret_bodies)} reviews on this page")
                
#                 # Process found reviews up to target_reviews
#                 for i in range(min(len(ret_bodies), target_reviews - len(reviews))):
#                     review_titles.append(ret_titles[i].text.strip())
#                     reviews.append(ret_bodies[i].text.strip())
#                     ratings.append(ret_ratings[i].text.strip())
                
#                 # Stop if we've reached our target
#                 if len(reviews) >= target_reviews:
#                     print(f"✅ Reached target of {target_reviews} reviews.")
#                     break
                    
#                 # Try to go to next page
#                 try:
#                     print("Looking for next page button...")
#                     next_button = WebDriverWait(driver, 10).until(
#                         EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
#                     )
#                     next_button.click()
#                     page_count += 1
#                     time.sleep(2)  # Brief pause after clicking
#                 except Exception as e:
#                     print(f"No more pages available: {str(e)}")
#                     break
                    
#             print(f"Scraping complete. Retrieved {len(reviews)} reviews.")
            
#         except Exception as e:
#             print(f"Error during scraping: {str(e)}")
#         finally:
#             driver.quit()
        
#         # Limit to target_reviews in case we got more
#         return review_titles[:target_reviews], reviews[:target_reviews], ratings[:target_reviews]


#     def _setup_driver(self):
#         """Set up the Selenium WebDriver with appropriate options"""
#         chrome_options = Options()
#         chrome_options.add_argument('--disable-gpu')
#         chrome_options.add_argument('--disable-dev-shm-usage')
#         chrome_options.add_argument('--no-sandbox')
#         chrome_options.add_argument('--log-level=3')  # Minimize logging
#         chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
#         chrome_options.add_experimental_option('useAutomationExtension', False)
#         prefs = {
#             "credentials_enable_service": False,         # Disables the password manager
#             "profile.password_manager_enabled": False    # Disables the password manager UI
#         }
#         chrome_options.add_experimental_option("prefs", prefs)
        
#         return webdriver.Chrome(options=chrome_options)

#     def _save_to_csv(self, titles, reviews, ratings, product_name="Amazon Product"):
#         """Save reviews to CSV in format compatible with preprocessing.ipynb"""
#         # Create a DataFrame
#         df = pd.DataFrame({
#             "id": range(1, len(titles) + 1),
#             "name": product_name,
#             "brand": "Amazon",
#             "categories": "Product",
#             "primaryCategories": "Product",
#             "reviews.doRecommend": [True if r.startswith("5") or r.startswith("4") else False for r in ratings],
#             "reviews.rating": [float(r.split()[0]) if r and r[0].isdigit() else 0 for r in ratings],
#             "reviews.text": reviews,
#             "reviews.title": titles
#         })
        
#         # Save as reviews_cleaned.csv for compatibility with existing code
#         compatibility_filename = "scraped_reviews.csv"
#         df.to_csv(compatibility_filename, index=False)
#         print(f"✅ Saved {len(df)} reviews to {compatibility_filename} for analysis")
        
#         return compatibility_filename

# Version 4: Enhanced with undetected_chromedriver

from typing import ClassVar, Optional, Dict, Any, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import time
import os
import random
import warnings
import pandas as pd
import re
from dotenv import load_dotenv
import logging
import sys

# Suppress the ChromeDriver cleanup warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
if sys.platform == 'win32':
    # Suppress Windows handle errors during cleanup
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='amazon_scraper.log')
logger = logging.getLogger('amazon_scraper')

# Define the input schema using Pydantic
class AmazonScraperSchema(BaseModel):
    url: str = Field(
        description="Amazon product URL to scrape reviews from"
    )
    target_reviews: int = Field(
        description="Number of reviews to scrape"
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Optional name to use for the product (default: extracted from page)"
    )

class AmazonScraperTool(BaseTool):
    """Tool that scrapes customer reviews from Amazon product pages"""

    name: str = "amazon_scraper_tool"
    description: str = "Scrapes customer reviews from Amazon product pages"
    args_schema: Type[BaseModel] = AmazonScraperSchema

    # Use class variables to store credentials
    _username: str = PrivateAttr()
    _password: str = PrivateAttr()

    def __init__(self):
        # First initialize the parent class
        super().__init__()

        # Load environment variables
        load_dotenv()
        self._username = os.getenv('NUMBER')
        self._password = os.getenv('PASSWORD')

        # Check if credentials are available
        if not self._username or not self._password:
            logger.error("Missing credentials in .env file")
            print("⚠️ Error: Missing credentials in .env file")
            print("Please ensure your .env file contains:")
            print("NUMBER=your_email_or_phone_number")
            print("PASSWORD=your_amazon_password")
        else:
            logger.info(f"Credentials loaded successfully (username: {self._username[:2]}...{self._username[-2:]})") 
            print(f"✓ Credentials loaded successfully (username: {self._username[:2]}...{self._username[-2:]})")
    
    def _random_sleep(self, a=0.4, b=1.0):
        """Short randomized delay to mimic human-like timing."""
        time.sleep(random.uniform(a, b))
    
    def _run(self, url: str, target_reviews: int, product_name: Optional[str] = None) -> str:
        """Run the tool with the given inputs."""
        print(f"DEBUG: Received URL to scrape: {url}")
    
        # Ensure we're keeping the original URL domain intact
        if "amazon.in" in url:
            print("Using Indian Amazon domain (amazon.in)")
        
        # Check for credentials
        if not self._username or not self._password:
            return "Error: Missing Amazon credentials in .env file"
        
        # Preserve the original URL - ensure we're not modifying it
        original_url = url
        logger.info(f"Starting to scrape {target_reviews} reviews from: {original_url}")
        print(f"Starting to scrape {target_reviews} reviews from: {original_url}")
        
        # Ensure URL is properly formatted (handle Amazon short URLs)
        if not re.match(r'^https?://', original_url):
            original_url = f"https://{original_url}"
        
        # Set up the driver and scrape reviews
        try:
            reviews_titles, reviews_texts, ratings = self._perform_scraping(url, target_reviews)
            
            # Save to CSV
            if not product_name:
                product_name = "Amazon Product"  # Default name if not provided
                
            csv_path = self._save_to_csv(reviews_titles, reviews_texts, ratings, product_name)
            
            return f"Successfully scraped {len(reviews_texts)} reviews. Data saved to {csv_path}"
        
        except Exception as e:
            error_msg = f"Error during scraping: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _perform_scraping(self, url: str, target_reviews):
        """Internal method to handle the actual scraping logic with undetected_chromedriver"""
        driver = None
        review_titles = []
        reviews = []
        ratings = []

        try:
            driver = self._setup_driver()
            logger.info(f"Opening URL: {url}")
            print(f"Opening URL: {url}")
            driver.get(url)

            # ---------------- OPEN REVIEWS PAGE ---------------- #
            try:
                WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
                ).click()
            except TimeoutException:
                # Amazon frequently omits or changes this link on consent,
                # regional, and anti-bot pages. Navigate directly when the
                # product ASIN is available.
                asin_match = re.search(r"/dp/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
                if not asin_match:
                    raise TimeoutException(
                        "Amazon reviews link was not found and the product URL has no 10-character ASIN"
                    )
                reviews_url = (
                    f"https://{url.split('/')[2]}/product-reviews/"
                    f"{asin_match.group(1)}"
                )
                logger.warning(
                    "Reviews link not found; navigating directly to %s", reviews_url
                )
                driver.get(reviews_url)

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

                logger.info(f"Logging in as: {self._username}")
                print(f"Logging in as: {self._username}")
                email_input.send_keys(self._username)
                self._random_sleep()

                continue_button = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.ID, "continue"))
                )
                continue_button.click()

                password_input = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "ap_password"))
                )
                password_input.send_keys(self._password)
                self._random_sleep()

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
                    logger.info("⚠️ Detected CAPTCHA or 2FA challenge.")
                    print("⚠️ Detected CAPTCHA or 2FA challenge.")
                    input("Please complete it manually in the browser, then press Enter to continue...")

                elif not login_ok:
                    logger.warning("⚠️ Login not confirmed, continuing anyway...")
                    print("⚠️ Login not confirmed, continuing anyway...")

            except Exception as e:
                logger.warning(f"⚠️ Login flow skipped or failed: {e}")
                print(f"⚠️ Login flow skipped or failed: {e}")

            # ---------------- POST-LOGIN NAVIGATION ---------------- #
            logger.info("Navigating to target URL...")
            print("Navigating to target URL...")
            try:
                try:
                    driver.get(url)
                except TimeoutException:
                    logger.warning("Timeout on driver.get(); using JS fallback navigation.")
                    print("Timeout on driver.get(); using JS fallback navigation.")
                    driver.execute_script("window.location.href = arguments[0];", url)
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Navigation error: {e}")
                print(f"Navigation error: {e}")
                driver.execute_script("window.location.href = arguments[0];", url)
                time.sleep(2)

            # Click reviews again after login
            try:
                WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
                ).click()
            except Exception as e:
                logger.warning(f"⚠️ Couldn't click reviews link after login: {e}")
                print(f"⚠️ Couldn't click reviews link after login: {e}")

            # ---------------- SCRAPING SECTION ---------------- #
            pages_to_scrape = target_reviews // 10 + 1  # Approximate pages needed (10 reviews per page)
            for page_num in range(pages_to_scrape):
                self._random_sleep(1, 2)

                soup = BeautifulSoup(driver.page_source, "html.parser")
                with open("amazon_reviews_page.txt", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)

                titles = soup.find_all("a", {"data-hook": "review-title"})
                bodies = soup.find_all("span", {"data-hook": "review-body"})
                stars = soup.find_all("span", class_="a-icon-alt")

                logger.info(f"Scraping page {page_num + 1}... Found {len(bodies)} reviews")
                print(f"Scraping page {page_num + 1}... Found {len(bodies)} reviews")

                for i in range(min(len(titles), len(bodies))):
                    review_titles.append(titles[i].text.strip())
                    reviews.append(bodies[i].text.strip())
                    ratings.append(stars[i].text.strip() if i < len(stars) else "")
                    
                    # Stop if we've reached target
                    if len(reviews) >= target_reviews:
                        logger.info(f"✅ Reached target of {target_reviews} reviews.")
                        print(f"✅ Reached target of {target_reviews} reviews.")
                        break
                
                # Check if we have enough reviews
                if len(reviews) >= target_reviews:
                    break

                # Go to next page if exists
                try:
                    next_btn = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((
                            By.XPATH,
                            '//li[@class="a-last"]/a'
                        ))
                    )
                    self._random_sleep(0.5, 1.0)
                    next_btn.click()
                except Exception:
                    logger.info("No more pages found or next button not clickable.")
                    print("No more pages found or next button not clickable.")
                    break

            logger.info(f"✅ Scraping completed successfully! Retrieved {len(reviews)} reviews.")
            print(f"✅ Scraping completed successfully! Retrieved {len(reviews)} reviews.")
            
        except Exception as e:
            logger.error("❌ Error during scraping: %s", e, exc_info=True)
            print(f"❌ Error during scraping: {type(e).__name__}: {e}")
            raise
        finally:
            if driver is not None:
                try:
                    # quit() owns both browser and driver-service shutdown. Calling
                    # service.stop() first leaves an invalid Windows process handle.
                    driver.quit()
                except OSError as e:
                    # Chrome may already have exited after a crash; do not hide
                    # the original scraping exception in that case.
                    if "WinError 6" not in str(e) and "handle is invalid" not in str(e).lower():
                        logger.warning("⚠️ Error closing driver: %s", e)
                except Exception as e:
                    logger.warning("⚠️ Error closing driver: %s", e)
        
        # Limit to target_reviews in case we got more
        return review_titles[:target_reviews], reviews[:target_reviews], ratings[:target_reviews]

    def _setup_driver(self):
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

    def _save_to_csv(self, titles, reviews, ratings, product_name="Amazon Product"):
        """Save reviews to CSV in format compatible with preprocessing.ipynb"""
        # Create a DataFrame
        df = pd.DataFrame({
            "id": range(1, len(titles) + 1),
            "name": product_name,
            "brand": "Amazon",
            "categories": "Product",
            "primaryCategories": "Product",
            "reviews.doRecommend": [True if r.startswith("5") or r.startswith("4") else False for r in ratings],
            "reviews.rating": [float(r.split()[0]) if r and r[0].isdigit() else 0 for r in ratings],
            "reviews.text": reviews,
            "reviews.title": titles
        })
        
        # Save as scraped_reviews.csv
        compatibility_filename = "scraped_reviews.csv"
        df.to_csv(compatibility_filename, index=False)
        logger.info(f"✅ Saved {len(df)} reviews to {compatibility_filename} for analysis")
        print(f"✅ Saved {len(df)} reviews to {compatibility_filename} for analysis")
        
        return compatibility_filename




# from typing import Optional
# from langchain.tools import tool
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from bs4 import BeautifulSoup
# import time
# import os
# import pandas as pd
# from dotenv import load_dotenv

# # Load environment variables once at module level
# load_dotenv()
# _USERNAME = os.getenv('NUMBER')
# _PASSWORD = os.getenv('PASSWORD')

# @tool
# def amazon_scraper_tool(url: str, target_reviews: int = 50, product_name: Optional[str] = None) -> str:
#     """Scrapes customer reviews from Amazon product pages.
    
#     Args:
#         url: Amazon product URL to scrape reviews from
#         target_reviews: Number of reviews to scrape (default: 50, max: 500)
#         product_name: Optional name to use for the product (default: extracted from page)
        
#     Returns:
#         Path to the CSV file containing the scraped reviews
#     """
#     # Check for credentials
#     if not _USERNAME or not _PASSWORD:
#         return "Error: Missing Amazon credentials in .env file"
    
#     print(f"Starting to scrape {target_reviews} reviews from: {url}")
    
#     # Set up the driver and scrape reviews
#     try:
#         reviews_titles, reviews_texts, ratings = _perform_scraping(url, target_reviews)
        
#         # Save to CSV
#         if not product_name:
#             product_name = "Amazon Product"  # Default name if not provided
            
#         csv_path = _save_to_csv(reviews_titles, reviews_texts, ratings, product_name)
        
#         return f"Successfully scraped {len(reviews_texts)} reviews. Data saved to {csv_path}"
    
#     except Exception as e:
#         return f"Error during scraping: {str(e)}"

# def _perform_scraping(url, target_reviews):
#     """Internal method to handle the actual scraping logic"""
#     driver = _setup_driver()
#     review_titles = []
#     reviews = []
#     ratings = []

#     try:
#         print(f"Opening URL: {url}")
#         driver.get(url)

#         # Wait and click on the "See all reviews" link
#         print("Looking for 'See all reviews' link...")
#         WebDriverWait(driver, 20).until(
#             EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
#         ).click()
        
#         # Wait for email input field and enter email
#         print("Logging in with provided credentials...")
#         email_input = WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.ID, "ap_email_login"))
#         )
#         print(f"Using username: {_USERNAME[:2]}...{_USERNAME[-2:]}")
#         email_input.send_keys(_USERNAME)
        
#         enter_button = WebDriverWait(driver, 20).until(
#             EC.element_to_be_clickable((By.ID, "continue"))
#         )
#         enter_button.click()
        
#         password_input = WebDriverWait(driver, 20).until(
#             EC.presence_of_element_located((By.ID, "ap_password"))
#         )
#         password_input.send_keys(_PASSWORD)
        
#         sign_in_button = WebDriverWait(driver, 20).until(
#             EC.element_to_be_clickable((By.ID, "signInSubmit"))
#         )
#         sign_in_button.click()
        
#         page_count = 1
#         max_pages = 10  # Maximum number of pages to scrape
        
#         # Continue until we reach target_reviews or max_pages
#         while len(reviews) < target_reviews and page_count <= max_pages:
#             print(f"Scraping page {page_count}...")
#             # Wait for reviews to load
#             time.sleep(5)  # Give more time for page to load
            
#             tables = driver.page_source
            
#             soup = BeautifulSoup(tables, 'html.parser')
#             ret_bodies = soup.find_all('span', {'data-hook': 'review-body'})
#             ret_titles = soup.find_all('a', {'data-hook': 'review-title'})  
#             ret_ratings = soup.find_all('span', class_='a-icon-alt')
            
#             print(f"Found {len(ret_bodies)} reviews on this page")
            
#             # Process found reviews up to target_reviews
#             for i in range(min(len(ret_bodies), target_reviews - len(reviews))):
#                 review_titles.append(ret_titles[i].text.strip())
#                 reviews.append(ret_bodies[i].text.strip())
#                 ratings.append(ret_ratings[i].text.strip())
            
#             # Stop if we've reached our target
#             if len(reviews) >= target_reviews:
#                 print(f"✅ Reached target of {target_reviews} reviews.")
#                 break
                
#             # Try to go to next page
#             try:
#                 print("Looking for next page button...")
#                 next_button = WebDriverWait(driver, 10).until(
#                     EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
#                 )
#                 next_button.click()
#                 page_count += 1
#                 time.sleep(2)  # Brief pause after clicking
#             except Exception as e:
#                 print(f"No more pages available: {str(e)}")
#                 break
                
#         print(f"Scraping complete. Retrieved {len(reviews)} reviews.")
        
#     except Exception as e:
#         print(f"Error during scraping: {str(e)}")
#     finally:
#         driver.quit()
    
#     # Limit to target_reviews in case we got more
#     return review_titles[:target_reviews], reviews[:target_reviews], ratings[:target_reviews]

# def _setup_driver():
#     """Set up the Selenium WebDriver with appropriate options"""
#     chrome_options = Options()
#     chrome_options.add_argument('--disable-gpu')
#     chrome_options.add_argument('--disable-dev-shm-usage')
#     chrome_options.add_argument('--no-sandbox')
#     chrome_options.add_argument('--log-level=3')  # Minimize logging
#     chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
#     chrome_options.add_experimental_option('useAutomationExtension', False)
#     prefs = {
#         "credentials_enable_service": False,         # Disables the password manager
#         "profile.password_manager_enabled": False    # Disables the password manager UI
#     }
#     chrome_options.add_experimental_option("prefs", prefs)
    
#     return webdriver.Chrome(options=chrome_options)

# def _save_to_csv(titles, reviews, ratings, product_name="Amazon Product"):
#     """Save reviews to CSV in format compatible with preprocessing.ipynb"""
#     # Create a DataFrame
#     df = pd.DataFrame({
#         "id": range(1, len(titles) + 1),
#         "name": product_name,
#         "brand": "Amazon",
#         "categories": "Product",
#         "primaryCategories": "Product",
#         "reviews.doRecommend": [True if r.startswith("5") or r.startswith("4") else False for r in ratings],
#         "reviews.rating": [float(r.split()[0]) if r and r[0].isdigit() else 0 for r in ratings],
#         "reviews.text": reviews,
#         "reviews.title": titles
#     })
    
#     # Save as reviews_cleaned.csv for compatibility with existing code
#     compatibility_filename = "reviews_cleaned.csv"
#     df.to_csv(compatibility_filename, index=False)
#     print(f"✅ Saved {len(df)} reviews to {compatibility_filename} for analysis")
    
#     return compatibility_filename