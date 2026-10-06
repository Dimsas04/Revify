from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time 

# Set up the Selenium driver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--log-level=3')  # Minimize logging
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    return webdriver.Chrome(options=chrome_options)

# Function to scrape the table and convert to DataFrame
def scrape_reviews(url):
    driver = setup_driver()
    data = []
    
    try:
        driver.get(url)
        time.sleep(20)  # Wait for the page to load
        
        # Use pandas to extract the main table
        tables = driver.page_source
        # print(tables)
        
        reviews_titles=[]
        reviews=[]
        soup = BeautifulSoup(tables, 'html.parser')
        spans = soup.find_all('a', class_='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold')
        for span in spans:
            title = span.find_all('span')[2].text.strip()  # Extract review title
            rating = span.find('span', class_='a-icon-alt').text.strip()  # Extract star rating
            reviews_titles.append((rating, title))
        
        
        # spans = soup.find_all('span', class_='a-size-base review-text')
        review_blocks = soup.find_all("span", {"data-hook": "review-body"})
        reviews = [block.find("span").text.strip() for block in review_blocks]
        return reviews_titles,reviews
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        

# Main function
def main():
    url = "https://www.amazon.in/ASIAN-Wonder-13-Sports-Running-Shoes/dp/B01N54ZM9W/ref=sr_1_5?dib=eyJ2IjoiMSJ9.RO0yaL3Zrti4xtgXRJZ65gmdWm2zF49TFp4N3tedFQCtNoS45e56xRfNbkPJLr78IlmrXGCYUDpzP7u957uyxfHKgeyLrRFnFzoC_Hjen-B-h0VfHQsB8Ni6sQ0Y1tHL2m10L-GL3wII0JhZBB3jqNHr4juFMGaEqldGEja6E41UgDpyaH7hs5K-DBABKQ0hPWgJzoQGMMv3wvqZUyiW_68LXdwHjm0J_UML_WRPEdogitcUX9dlx6sQ_I9tKEHlf5glQ2I1gd5Nh_Txx6B8EfN33o5Xpx8bKYm0Vcu8ZmVQVM2315OgNjAxcC8P0sdDZP7S2Ng-VfqGbWIEDaYJcz2lDMg_gSQpEZLdxSaiwYK3RhaqUeIS9p-JDeX8_OPVb5R4BrcqgCaBt_asVIokk5jkt_00Cm1RT1cPsJKHDSSaVGvxOpwrbT8I7LOSKTtW.X1ATNI550ekpXTdYXr3YBCZZizQT7uqm6KHnLZqUZFg&dib_tag=se&keywords=shoes&qid=1739688983&sr=8-5&th=1&psc=1"
    reviews_titles,reviews = scrape_reviews(url)
    
    for i in range(len(reviews_titles)):
        print(reviews_titles[i])
        print(reviews[i])
        print("\n")

if __name__ == "__main__":
    main()