import random
import re
import pandas as pd
from Lang_Detector import LangDetect
from Crawler_YT import Crawling_YT


class Main:

    # Creating an instance of class LangDetect
    lang_det = LangDetect()
    # Creating an instance of class Crawling_YT
    my_crawler = Crawling_YT()

    # Reading a training dataset with sentences
    txt_file = 'exported_files\dataset_sentences.txt'
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
    dataset_df.to_html('exported_files\gold.html')
    dataset_df.to_csv('exported_files\gold.csv')

    # Comparison of ground truth with my classification via regexps
    my_accuracy = lang_det.comp_scores(dataset_df, 'language', 'ground_truth')

    url_youtube = 'https://www.youtube.com/watch?v=b0z_dp5-luQ'

    # Lets store 15 or more Greek/Greeklish titles and their comments
    accepted_videos = {}
    result1 = my_crawler.crawl_yt_title(url_youtube)
    if result1 is not None:
        title1, comments1 = result1
        accepted_videos[title1] = url_youtube
    else:
        accepted_videos['My first video title'] = url_youtube
    check_next_videos = my_crawler.crawl_next(url_youtube)
    sum = 0
    loops = 0
    while sum < 20:
        loops += 1
        for key, value in check_next_videos.items():
            title_lang = lang_det.comp_languages(key, lang_choices)
            if title_lang == lang_choices[0] or title_lang == lang_choices[1]:
                accepted_videos[key] = value
        sum = len(accepted_videos)
        if loops < 10:
            random_key = random.choice(list(check_next_videos.keys()))
            random_url = check_next_videos[random_key]
            check_next_videos = my_crawler.crawl_next(random_url)
        else:
            check_next_videos = my_crawler.crawl_next(url_youtube)

    comments_list = []
    com_columns = []
    comments_df = pd.DataFrame(columns=['comment', 'link'])
    comments_df = comments_df.set_index('comment')

    for key, value in accepted_videos.items():
        result = my_crawler.crawl_yt_title(value)
        if result is not None:
            title, comments = result
            for comment in comments:
                comments_df.loc[comment] = value

    comments_df.to_csv('exported_files\crawl.csv')
    comments_df.to_html('exported_files\crawl.html')






