from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
import os
from amazoncaptcha import AmazonCaptcha
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
def scrape_reviews(url):
    driver = setup_driver()
    data = []

    driver.get(url)
    
    captcha_img = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div/div[1]/div[3]/div/div/form/div[1]/div/div/div[1]/img"))
    )
    img_src = captcha_img.get_attribute('src')
    driver.execute_script(f"window.open('{img_src}', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(1)
    with open('captcha.jpeg', 'wb') as file:
        file.write(driver.find_element(By.TAG_NAME, 'img').screenshot_as_png)
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    captcha = AmazonCaptcha('captcha.jpeg')
    solution = captcha.solve()
    
    captcha_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div/div[1]/div[3]/div/div/form/div[1]/div/div/div[2]/input"))
    )
    captcha_input.send_keys(solution)
    captcha_input.submit()
    

        # Wait and click on the "See all reviews" link
    WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="reviews-medley-footer"]/div[2]/a'))
    ).click()
    
    # time.sleep(3000)
    # Wait for the captcha image and download it
    # Wait for email input field and enter email
    email_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "ap_email_login"))
    )
    print(username)
    email_input.send_keys(username) #idhar apna email daalna hai
    enter_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.ID, "continue"))
    )
    enter_button.click()
    
    password_input = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "ap_password"))
    )
    password_input.send_keys(password)#idhar password
    
    sign_in_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.ID, "signInSubmit"))
    )
    sign_in_button.click()
    
    # time.sleep(5000)
    a=7
    review_titles = []
    reviews = []
    ratings=[]
    while(a!=0):
        a-=1
        time.sleep(2)


        tables = driver.page_source

        soup = BeautifulSoup(tables, 'html.parser')
        with open('amazon_reviews_page.txt', 'w', encoding='utf-8') as f:
            f.write(tables)
        ret_bodies = soup.find_all('span', {'data-hook': 'review-body'})
        ret_titles = soup.find_all('a', {'data-hook': 'review-title'})  
        ret_ratings= soup.find_all('span', class_='a-icon-alt')  
        
        # print(len(ret_bodies))
        # print(len(ret_titles))
        # print(len(ret_ratings))
        
        for i in range(0, len(ret_bodies)):
            # timestamp = date.text.strip()
            # print(timestamp)
            # reviews.append(timestamp)
            review_titles.append(ret_titles[i].text.strip())
            reviews.append(ret_bodies[i].text.strip())
            ratings.append(ret_ratings[i].text.strip())
            
            # print(ret_titles[i].text.strip())
            # print(ret_bodies[i].text.strip())
            # print(ret_ratings[i].text.strip())
            
            # print("\n\n")
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[2]/div/div[1]/div/div[1]/div[5]/div[3]/div/div/span/div/ul/li[2]/a'))
            )
            next_button.click()
        except:
            break
    return review_titles, reviews, ratings



# Main function
def main():
    url = "https://www.amazon.in/ASIAN-Wonder-13-Sports-Running-Shoes/dp/B01N54ZM9W/ref=sr_1_5"
    reviews_titles, reviews,ratings = scrape_reviews(url)

    for i in range(len(reviews_titles)):
        print(reviews_titles[i])
        # print(ratings[i])
        print(reviews[i])
        print("\n\n")

if __name__ == "__main__":
    main()

