import time
import requests
from selenium.common import WebDriverException
from selenium.webdriver import Edge, EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


class Crawling_YT:

    def crawl_yt_title(self, url):
        """With Edge as web driver i try to catch the title of the
        video and its comments. I implement that by searching the element with
         the correct XPATH for the title, and all the elements with the correct
          XPATH for the comments.  I use an automation in order to scroll down, so the necessary
          javascript code can be loaded smoothly. Moreover, I print if we have comments or not,
          thus I know when we have any problem (with the exceptions). I run an if condition in
          order to return a none value or the comments, title"""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                options = EdgeOptions()
                options.add_argument("--headless")
                with Edge(options=options) as driver:
                    driver.get(url)
                    wait = WebDriverWait(driver, 20)
                    title = wait.until(
                        EC.presence_of_element_located((By.XPATH, '//*[@id="title"]/h1/yt-formatted-string'))).text
                    for _ in range(5):
                        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
                        time.sleep(2)
                        comments = wait.until(
                            EC.presence_of_all_elements_located((By.XPATH, '//*[@id="comment-content"]')))
                        dates = wait.until(EC.presence_of_all_elements_located(
                            (By.XPATH, '//*[@id="header-author"]/yt-formatted-string')))
                    comment_list = []
                    date_list = []
                    for comment in comments:
                        comment_list.append(comment.text)
                    for date in dates:
                        date_list.append(date.text)

                if comment_list is not None:
                    print("Storing comments...")
                    return title, comment_list, date_list
                else:
                    print('No comments. Error follows...')
                    return None
            else:
                print(f"Problem with the connection. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"An exception occurred while making the request: {e}")

        except WebDriverException as e:
            print(f"An exception occurred with the WebDriver:\n No available comments!")

    def crawl_next(self, url):
        """With Edge as web driver I try to find as many as possible recommended videos
        in which the key will be the title and the value will be the link. I implement that by
         searching all the elements with the correct XPATH for the title, and all the elements
         with the correct XPATH for the link. I use an automation in order to scroll down,
         so the necessary javascript code can be loaded smoothly"""
        my_dict = {}
        try:
            response = requests.get(url)
            if response.status_code == 200:
                options = EdgeOptions()
                options.add_argument("--headless")
                with Edge(options=options) as driver:
                    driver.get(url)
                    wait = WebDriverWait(driver, 15)
                    for _ in range(2):
                        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
                        time.sleep(2)
                        next_titles = wait.until(
                            EC.presence_of_all_elements_located((By.XPATH, '//*[@id="dismissible"]/div/div[1]/a/h3')))
                        next_ids = wait.until(EC.presence_of_all_elements_located(
                            (By.XPATH, '//*[@id="dismissible"]/div/div[1]/a')))
                        for i in range(len(next_titles)):
                            my_dict[next_titles[i].text] = next_ids[i].get_attribute('href')

                return my_dict
            else:
                print(f"Problem with the connection. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"An exception occurred while making the request: {e}")

        except WebDriverException as e:
            print(f"An exception occurred with the WebDriver: {e}")

