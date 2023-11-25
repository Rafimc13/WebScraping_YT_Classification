import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver import Edge
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class LangDetect:

    def pattern_search(self, data, pattern):
        try:
            for i in range(len(data)):
                if re.search(pattern, data[i]):
                    data[i] = re.sub(pattern, "", data[i])
        except TypeError as e:
            print(f"You have a type error: {e}")

        return data

    @staticmethod
    def comp_gr_eng(dataset, pattern_gr, pattern_eng, languages, column='language'):
        for sentence in dataset.index:
            if re.search(pattern_gr, sentence):
                dataset.loc[sentence, column] = languages[0]
            elif re.search(pattern_eng, sentence):
                count_eng = len(re.findall(pattern_eng, sentence))
                my_pattern = re.compile(r'[!@#$%^&*()-,.?;:"]')
                sentence_test = re.sub(my_pattern, "", sentence)
                sentence_test = sentence_test.strip()
                sentence_test = sentence_test.replace(" ", "")
                if count_eng < 0.99 * len(sentence_test):
                    dataset.loc[sentence, column] = languages[3]
                else:
                    dataset.loc[sentence, column] = languages[2]

        return dataset

    def catch_greeklish(self, dataframe, patterns, language='greeklish'):
            pass

    @staticmethod
    def comp_scores(dataset, column1, column2):
        sum = 0
        for i in range(len(dataset)):
            if dataset.iloc[i][column1] == dataset.iloc[i][column2]:
                sum += 1
        score = sum/len(dataset)
        print(f'Accuracy of my language detector is: {score*100}%')
        return score


class Main:

    lang_det = LangDetect()

    ls_sentences = []
    with open('dataset_sentences.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line.strip()
            ls_sentences.append(line)

    pattern1 = re.compile('\n')
    pattern2 = re.compile(r'\d{1,2}\. ')

    ls_sentences = lang_det.pattern_search(ls_sentences, pattern1)
    ls_sentences = lang_det.pattern_search(ls_sentences, pattern2)

    my_columns = ['sentence', 'language', 'ground_truth', 'author']

    dataset_df = pd.DataFrame(columns=my_columns)
    dataset_df['sentence'] = ls_sentences
    dataset_df = dataset_df.set_index('sentence')

    dataset_df['ground_truth'][:30] = "english"
    dataset_df['ground_truth'][30:40] = "greek"
    dataset_df['ground_truth'][40:50] = "greeklish"
    dataset_df['ground_truth'][50:60] = "other"
    dataset_df['ground_truth'][60:70] = "other"
    dataset_df['ground_truth'][70:] = "other"
    dataset_df['author'] = 'chatGPT'

    pattern_greek = re.compile(r'[α-ωΑ-Ωίϊΐόύϋΰέώ]{3}')
    pattern_english = re.compile(r'[a-zA-Z]')
    lang_choices = ['greek', 'greeklish', 'english', 'other']

    dataset_df = lang_det.comp_gr_eng(dataset_df, pattern_greek, pattern_english, lang_choices)

    pattern_greeklish = re.compile(r'\b(gia|kai|ti|ta|tis|tous|tou|toys|oi|o|h|autos|auta|'
                                   r'oti|na|nai|oxi|den|tha|8a)\b', flags=re.IGNORECASE)

    for sentence in dataset_df[dataset_df['language'].isin(lang_choices[2:])].index:
        if re.search(pattern_greeklish, sentence):
            dataset_df.loc[sentence, 'language'] = lang_choices[1]

    dataset_df.to_html("gold.html")
    my_accuracy = lang_det.comp_scores(dataset_df, 'language', 'ground_truth')


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

