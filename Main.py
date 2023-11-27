import random
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

    # The language choices based on the exercise terms
    lang_choices = ['greek', 'greeklish', 'english', 'other']

    # Classification of the sentences via my function comp_languages
    for sentence in dataset_df.index:
        dataset_df.loc[sentence, 'language'] = lang_det.comp_languages(sentence, lang_choices)

    # Saving our dataframe in a .html and a .csv file with the name 'gold'
    dataset_df.to_html('exported_files\gold.html')
    dataset_df.to_csv('exported_files\gold.csv')

    # Comparison of ground truth with my classification via regexps
    my_accuracy = lang_det.comp_scores(dataset_df, 'language', 'ground_truth')

    # Take an initial video of youtube
    url_youtube = 'https://www.youtube.com/watch?v=b0z_dp5-luQ'

    # Storing 20 or more Greek/Greeklish titles and their links
    accepted_videos = {}
    result1 = my_crawler.crawl_yt_title(url_youtube) # It might not catch any comments thus I run an if condition
    if result1 is not None:
        title1, comments1 = result1
        accepted_videos[title1] = url_youtube
    else:
        accepted_videos['My first video title'] = url_youtube
    check_next_videos = my_crawler.crawl_next(url_youtube)
    sum, loops = 0, 0
    while sum < 20:  # Iterating a sum in order to catch at least 20 different videos
        loops += 1  # Loops if the crawler jumps without finding any 'correct' title to start from the initial url
        for key, value in check_next_videos.items():
            title_lang = lang_det.comp_languages(key, lang_choices)  # Check if lang of each title is Greek/Greeklish
            if title_lang == lang_choices[0] or title_lang == lang_choices[1]:
                accepted_videos[key] = value  # Save it in a dict of accepted videos
        sum = len(accepted_videos)
        if loops < 15: # if crawler loop 15 times start from the beginning
            random_key = random.choice(list(check_next_videos.keys()))
            random_url = check_next_videos[random_key]
            check_next_videos = my_crawler.crawl_next(random_url)
        else:
            check_next_videos = my_crawler.crawl_next(url_youtube)

    # Creating a dataframe in order to store our comments with their link
    comments_df = pd.DataFrame(columns=['comment', 'link'])
    comments_df = comments_df.set_index('comment')

    # Iterate the accepted link based on the title names and crawl their comments. Store them in the dataframe
    for key, value in accepted_videos.items():
        result = my_crawler.crawl_yt_title(value)
        if result is not None:
            title, comments = result
            for comment in comments:
                comments_df.loc[comment] = value

    # Store the dataframe in a .csv and a .html file with the name 'crawl'
    comments_df.to_csv('exported_files\crawl.csv')
    comments_df.to_html('exported_files\crawl.html')






