import re
from bs4 import BeautifulSoup
from selenium.webdriver import Edge
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class Crawling_YT:

    url_youtube = 'https://www.youtube.com/watch?v=b0z_dp5-luQ'

    with Edge() as driver:
        driver.get(url_youtube)
        wait = WebDriverWait(driver, 30)

        title = wait.until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="title"]/h1/yt-formatted-string'))).text
        print(title)
        title_next = wait.until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="dismissible"]/div/div[1]/a/h3'))).text
        id_next = wait.until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="dismissible"]/div/div[1]/a')))

        next_video_url = id_next.get_attribute("href")
        print(title_next)
        print(next_video_url)