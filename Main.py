import re
import pandas as pd
from Lang_Detector import LangDetect
from Crawler_YT import Crawling_YT

class Main:

    # Creating an instance of class LangDetect
    lang_det = LangDetect()

    my_crawler = Crawling_YT()

    # Reading a training dataset with sentences
    txt_file = 'dataset_sentences.txt'
    ls_sentences = lang_det.read_txt(txt_file)

    # Lets clean our dataset with the unnecessary staff (numbers, dots, etc)
    for i in range(len(ls_sentences)):
        ls_sentences[i] = lang_det.pattern_search(ls_sentences[i])

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

    lang_choices = ['greek', 'greeklish', 'english', 'other']

    for sentence in dataset_df.index:
        dataset_df.loc[sentence, 'language'] = lang_det.comp_languages(sentence, lang_choices)

    # Saving our dataframe in a .html and a .csv file with the name gold
    dataset_df.to_html("gold.html")
    dataset_df.to_csv('gold.csv')
    # Comparison of ground truth with my classification via regexps
    my_accuracy = lang_det.comp_scores(dataset_df, 'language', 'ground_truth')

    url_youtube = 'https://www.youtube.com/watch?v=ryE8V8iyH5c'

    accepted_list = []
    first_title = my_crawler.crawl_yt_title(url_youtube)
    accepted_list.append({first_title: url_youtube})
    accepted_list.append(my_crawler.crawl_next(url_youtube))

    print(accepted_list)