from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
import os
import sys
import pandas as pd

# Load environment variables
load_dotenv()
username = os.getenv('NUMBER')
password = os.getenv('PASSWORD')

# Check if credentials are available
if not username or not password:
    print("⚠️ Error: Missing credentials in .env file")
    print("Please ensure your .env file contains:")
    print("NUMBER=your_email_or_phone_number")
    print("PASSWORD=your_amazon_password")
    sys.exit(1)
else:
    print(f"✓ Credentials loaded successfully (username: {username[:2]}...{username[-2:]})")

# Set up the Selenium driver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--log-level=3')  # Minimize logging
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_experimental_option("detach", True)  # Keep browser open after script finishes
    prefs = {
        "credentials_enable_service": False,         # Disables the password manager
        "profile.password_manager_enabled": False    # Disables the password manager UI
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    return webdriver.Chrome(options=chrome_options)

# Function to scrape the table and convert to DataFrame
def scrape_reviews(url, target_reviews):
    driver = setup_driver()
    review_titles = []
    reviews = []
    ratings = []

    try:
        print(f"Opening URL: {url}")
        driver.get(url)

        # Wait and click on the "See all reviews" link
        print("Looking for 'See all reviews' link...")
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
        ).click()
        
        # Wait for email input field and enter email
        print("Logging in with provided credentials...")
        email_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "ap_email_login"))
        )
        print(f"Using username: {username[:2]}...{username[-2:]}")
        email_input.send_keys(username)
        
        enter_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "continue"))
        )
        enter_button.click()
        
        password_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "ap_password"))
        )
        password_input.send_keys(password)
        
        sign_in_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "signInSubmit"))
        )
        sign_in_button.click()
        
        page_count = 1
        max_pages = 10  # Maximum number of pages to scrape to avoid excessive scraping
        
        # Continue until we reach target_reviews or max_pages
        while len(reviews) < target_reviews and page_count <= max_pages:
            print(f"Scraping page {page_count}...")
            # Wait for reviews to load
            time.sleep(5)  # Give more time for page to load
            
            tables = driver.page_source
            
            soup = BeautifulSoup(tables, 'html.parser')
            ret_bodies = soup.find_all('span', {'data-hook': 'review-body'})
            ret_titles = soup.find_all('a', {'data-hook': 'review-title'})  
            ret_ratings = soup.find_all('span', class_='a-icon-alt')
            
            print(f"Found {len(ret_bodies)} reviews on this page")
            
            # Process found reviews up to target_reviews
            for i in range(min(len(ret_bodies), target_reviews - len(reviews))):
                review_titles.append(ret_titles[i].text.strip())
                reviews.append(ret_bodies[i].text.strip())
                ratings.append(ret_ratings[i].text.strip())
            
            # Stop if we've reached our target
            if len(reviews) >= target_reviews:
                print(f"✅ Reached target of {target_reviews} reviews.")
                break
                
            # Try to go to next page
            try:
                print("Looking for next page button...")
                # Using a more reliable selector for the next button
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "li.a-last a"))
                )
                next_button.click()
                page_count += 1
                time.sleep(2)  # Brief pause after clicking
            except Exception as e:
                print(f"No more pages available: {str(e)}")
                break
                
        print(f"Scraping complete. Retrieved {len(reviews)} reviews.")
        
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
    
    # Limit to target_reviews in case we got more
    return review_titles[:target_reviews], reviews[:target_reviews], ratings[:target_reviews]

# Save reviews to CSV in format compatible with your preprocessing.ipynb
def save_to_csv(titles, reviews, ratings, product_name="Amazon Product"):
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
    
    # Save to CSV
    filename = "scraped_reviews.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} reviews to {filename}")
    return df

# Main function
def main():
    url = "https://www.amazon.in/ASIAN-Wonder-13-Sports-Running-Shoes/dp/B01N54ZM9W/ref=sr_1_5"
    product_name = "ASIAN Wonder 13 Sports Running Shoes"
    
    # Scrape 30 reviews
    print(f"Starting to scrape 30 reviews for {product_name}...")
    reviews_titles, reviews, ratings = scrape_reviews(url, target_reviews=70)
    
    # Save to CSV in the format needed for your preprocessing script
    save_to_csv(reviews_titles, reviews, ratings, product_name)

    # Print a sample of scraped reviews
    print("\n=== Sample of scraped reviews ===")
    for i in range(min(3, len(reviews_titles))):
        print(f"Title: {reviews_titles[i]}")
        print(f"Rating: {ratings[i]}")
        print(f"Review: {reviews[i][:100]}..." if len(reviews[i]) > 100 else f"Review: {reviews[i]}")
        print("\n")
    
    print(f"Total reviews scraped: {len(reviews)}")

if __name__ == "__main__":
    main()