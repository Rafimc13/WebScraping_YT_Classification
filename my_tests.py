import time
from selenium.webdriver import Edge
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from Lang_Detector import LangDetect

url_youtube = 'https://www.youtube.com/watch?v=b0z_dp5-luQ'
video_comments = {}
with Edge() as driver:
    driver.get(url_youtube)
    wait = WebDriverWait(driver, 20)
    title = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="title"]/h1/yt-formatted-string'))).text
    for _ in range(5):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)
        comments = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="comment-content"]')))
    my_list = []
    for comment in comments:
        my_list.append(comment.text)


video_comments[title] = my_list
print(video_comments)