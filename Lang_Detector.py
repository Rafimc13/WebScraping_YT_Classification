import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver import Edge
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class LangDetect:


    def read_txt(self, txt):
        ls_sentences = []
        with open(txt, 'r', encoding='utf-8') as file:
            for line in file:
                line.strip()
                ls_sentences.append(line)

        return ls_sentences

    def pattern_search(self, sentence, pattern):
        try:
            sentence = re.sub(pattern, "", sentence)
        except Exception as e:
            print(f"You have an error: {e}")

        return sentence


    def comp_languages(self, sentence, pattern_gr, pattern_eng, pattern_greeklish, languages):
        lang = None
        if re.search(pattern_gr, sentence):
            lang = languages[0]
        elif re.search(pattern_eng, sentence):
            count_eng = len(re.findall(pattern_eng, sentence))
            my_pattern = re.compile(r'[!@#$%^&*(-),.?;:"><]')
            clean_sentence = self.pattern_search(sentence, my_pattern)
            clean_sentence = clean_sentence.replace(" ", "")
            if count_eng < 0.99 * len(clean_sentence):
                lang = languages[3]
            else:
                lang = languages[2]

        if lang == languages[2] or lang==languages[3]:
            if re.search(pattern_greeklish, sentence):
                lang = languages[1]

        return lang


    def comp_scores(self, dataset, column1, column2):
        sum = 0
        for i in range(len(dataset)):
            if dataset.iloc[i][column1] == dataset.iloc[i][column2]:
                sum += 1
        score = sum/len(dataset)
        print(f'Accuracy of my language detector is: {score*100}%')
        return score


class Main:

    # Creating an instance of class LangDetect
    lang_det = LangDetect()

    # Reading a training dataset with sentences
    txt_file = 'dataset_sentences.txt'
    ls_sentences = lang_det.read_txt(txt_file)

    # Lets clean our dataset with the unnecessary staff (numbers, dots, etc)
    pattern1 = re.compile(r'\.\n|(\d{1,2}\. )')
    for i in range(len(ls_sentences)):
        ls_sentences[i] = lang_det.pattern_search(ls_sentences[i], pattern1)

    # Create a dataframe in order to store our new data
    my_columns = ['sentence', 'language', 'ground_truth', 'author']
    dataset_df = pd.DataFrame(columns=my_columns)
    dataset_df['sentence'] = ls_sentences
    dataset_df = dataset_df.set_index('sentence')

    # Creating a ground truth evaluation dataset
    dataset_df['ground_truth'][:30] = "english"
    dataset_df['ground_truth'][30:40] = "greek"
    dataset_df['ground_truth'][40:50] = "greeklish"
    dataset_df['ground_truth'][50:] = "other"
    dataset_df['author'] = 'chatGPT'

    # Separating the Greek/English/other languages. If we have at least 3 greek chars is a greek title-comment
    pat_gr = re.compile(r'[α-ωΑ-Ωίϊΐόύϋΰέώ]{3}')
    pat_eng = re.compile(r'[a-zA-Z]')
    pat_greeklish = re.compile(r'\b(gia|kai|ti|ta|tis|tous|tou|toys|oi|o|h|autos|auta|'
                                   r'oti|na|nai|oxi|den|tha|8a)\b', flags=re.IGNORECASE)
    lang_choices = ['greek', 'greeklish', 'english', 'other']

    for sentence in dataset_df.index:
        dataset_df.loc[sentence, 'language'] = lang_det.comp_languages(sentence, pat_gr, pat_eng, pat_greeklish, lang_choices)

    dataset_df.to_html("gold.html")
    dataset_df.to_csv('gold.csv')
    my_accuracy = lang_det.comp_scores(dataset_df, 'language', 'ground_truth')


