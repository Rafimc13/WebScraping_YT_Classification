import re
from bs4 import BeautifulSoup
from selenium.webdriver import Edge
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from Lang_Detector import LangDetect


class Crawling_YT:

    def crawl_yt_title(self, url):

        with Edge() as driver:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            title = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="title"]/h1/yt-formatted-string'))).text

        return title

    def crawl_next(self, url):

        lang_choices = ['greek', 'greeklish', 'english', 'other']
        with Edge() as driver:
            lang_det = LangDetect()

            driver.get(url)
            wait = WebDriverWait(driver, 15)

            title_next = wait.until(EC.presence_of_element_located(
                (By.XPATH, '//*[@id="dismissible"]/div/div[1]/a/h3'))).text

            title_next = lang_det.pattern_search(title_next)
            id_next = wait.until(EC.presence_of_element_located(
                (By.XPATH, '//*[@id="dismissible"]/div/div[1]/a')))
            next_video_url = id_next.get_attribute("href")
            lang = lang_det.comp_languages(title_next, lang_choices)
        if lang == lang_choices[0] or lang == lang_choices[1]:
            accepted_dict = {title_next: next_video_url}
            return accepted_dict
        else:
            self.crawl_next(next_video_url)









        #


