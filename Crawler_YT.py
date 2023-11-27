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
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # ChatGPT line of code
                        time.sleep(2)
                        comments = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="comment-content"]')))
                    comment_list = []
                    for comment in comments:
                        comment_list.append(comment.text)
                if comment_list is not None:
                    print("Storing comments...")
                    return title, comment_list
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
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # ChatGPT line of code
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

