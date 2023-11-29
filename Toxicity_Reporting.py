import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from Classification import ClassificationTrain as clt
from GPT_Prompting import PromptingGPT


class Generate_Toxicity:

    def combine_dict(self, df, dict1, dict2):
        for key, value in dict1.items():
            dict1[key] = value + dict2[key]

        df.loc[dict1['class 1 (non-toxic)'], 'toxicity'] = 1
        df.loc[dict1['class 2'], 'toxicity'] = 2
        df.loc[dict1['class 3'], 'toxicity'] = 3
        df.loc[dict1['class 4'], 'toxicity'] = 4
        df.loc[dict1['class 5 (fully toxic)'], 'toxicity'] = 5

        return dict1

    def most_toxic_lang(self, df):
        gr_toxic = []
        eng_toxic = []
        other_toxic = []
        for i in range(len(df)):
            if df.loc[i, 'language'] == 'greek':
                gr_toxic.append(int(df.loc[i, 'toxicity']))
            elif df.loc[i, 'language'] == 'english':
                eng_toxic.append(int(df.loc[i, 'toxicity']))
            else:
                other_toxic.append(int(df.loc[i, 'toxicity']))

        gr_count = (gr_toxic.count(3) + gr_toxic.count(4) + gr_toxic.count(5)) / len(gr_toxic)
        eng_count = (eng_toxic.count(3) + eng_toxic.count(4) + eng_toxic.count(5)) / len(eng_toxic)
        other_count = (other_toxic.count(3) + other_toxic.count(4) + other_toxic.count(5)) / len(other_toxic)

        if gr_count > eng_count and gr_count > other_count:
            print(f'The most toxic language is: Greek, with percentage {gr_count*100:.3f}% toxicity')
        elif eng_count > gr_count and eng_toxic > other_count:
            print(f'The most toxic language is: English, with percentage {eng_count*100:.3f}% toxicity')
        else:
            print(f'The most toxic language is: other languages, with percentage {other_count*100:.3f}% toxicity')

    def highest_toxic_page(self):
        pages_set = set()

        for link in df.loc[:, 'link']:
            pages_set.add(link)

        page_toxicity_dict = {}
        for page in pages_set:
            toxicities_for_page = df.loc[df['link'] == page, 'toxicity'].tolist()
            page_toxicity_dict[page] = toxicities_for_page
        best_count = -1
        most_toxic_page = None
        for page in pages_set:
            page_count = page_toxicity_dict[page]
            toxicity_count = (page_count.count(3) + page_count.count(4) + page_count.count(5)) / len(page_count)
            if toxicity_count >= best_count:
                best_count = toxicity_count
                most_toxic_page = page

        print(
            f'The page with the highest rate of toxic posts is: {most_toxic_page}, and its rate is: {best_count * 100:.2f}%')
        return pages_set

    def convert_greek_date(self, greek_date):
        match = re.search(r'πριν από (\d+) (ημέρες|μήνες|έτη)', greek_date)
        if match:
            value = int(match.group(1))
            unit = match.group(2)

            if unit == 'ημέρες':  # chatGPT lines of code
                return pd.Timedelta(days=value)
            elif unit == 'μήνες':
                return pd.Timedelta(days=value * 30)
            elif unit == 'έτη':
                return pd.Timedelta(days=value * 365)

        return pd.NaT

    def toxic_over_time(self, df, pages_set):
        datetime_page = {}
        for page in pages_set:
            dates_for_page = df.loc[df['link'] == page, 'date'].tolist()
            dates_for_page = [date for date in dates_for_page if not pd.isna(date)]  # ChatGPT line of code
            dates_for_page.sort()
            datetime_page[page] = dates_for_page

        i = 0
        # Assuming datetime_page is your dictionary of sorted dates for each page
        for page, dates in datetime_page.items():
            i += 1
            if i == 1:
                plt.figure(figsize=(12, 6))
            toxicities = df.loc[
                (df['link'] == page) & (~df['date'].isna()), 'toxicity'].tolist()  # ChatGPT line of code
            plt.subplot(220 + i)
            dates_numeric = date2num(dates)
            plt.plot(dates_numeric, toxicities, label=page)
            plt.xlabel('Date')
            plt.ylabel('Toxicity')
            plt.title('Toxicity Over Time for Each Page')
            plt.legend()

            if i == 4:
                plt.tight_layout()
                plt.show()
                i = 0

        if i != 0:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    GPT_prompts = PromptingGPT()
    toxic = Generate_Toxicity()

    df = pd.read_csv('exported_files/crawl.csv', index_col=0)
    training_comments1 = df['comment'][:50].to_string()
    training_comments2 = df['comment'][50:80].to_string()

    prompt1 = ("Hello, I would like you to  write me 50 new comments in Greek. Each comment is on a "
               "scale from 1 (non-toxic) to 5 (fully toxic). "
               "So, write me these 50 comments randomly and please I would like each class to have the same number of comments."
               "For this purpose you are provided with a set of example comments "
               "labeled with their corresponding toxicity scores. Note that I want 50 new comments in Greek. "
               "These are only examples in order to help you give me the 50 new comments. Thank you!"
               "Examples:"
               "Σχόλιο: Εκτιμώ την σκεπτική συζήτηση σε αυτό το νήμα."
               "Βαθμός Δηλητηρίασης: 1"

               "Σχόλιο: Παρά την διαφωνία με το όραμά σας, ας διατηρήσουμε τον σεβασμό στη συζήτηση."
               "Βαθμός Δηλητηρίασης: 2"

               "Σχόλιο: Η επιχειρηματολογία σας φαίνεται αδύναμη και έχει έλλειψη αξιόπιστων επιχειρημάτων."
               "Βαθμός Δηλητηρίασης: 3"

               "Σχόλιο: Είσαι απόλυτα ιδιώτης που και που προτείνει τέτοιες ανοησίες."
               "Βαθμός Δηλητηρίασης: 4"

               "Σχόλιο: Ελπίζω να υποφέρεις για τις ανόητες απόψεις σου."
               "Βαθμός Δηλητηρίασης: 5")

    prompt2 = ('So based on the above comments and classification, I would like to classify the below comments'
               'from 1 (non-toxic)to 5 (fully toxic). So, you will give me 5 classes. Please provide me the output '
               'via a dictionary format. The key will be the class and the value will be a list of the integers that are'
               'in the beginning of each sentence. So the sentence 0 Ο "Έκπτωτος Άγγελος" μόλις κυκλοφόρησε και είμ... '
               'will be assigned in a class as 0.'
               'For example:'
               'class 1 (non-toxic): [1,7,13 etc]') + training_comments1

    prompt3 = ('Moreover based on the above comments and classification, I would like to classify with the same '
               'pattern from 1 (non-toxic)to 5 (fully toxic) the below sentences. I will give you again an example'
               'For example:'
               'class 1 (non-toxic): [1,7,13 etc]') + training_comments2

    # I comment on these 3 prompts because it demands payment for the API of GPT - Cost for these prompts (three times run): 0.09euro
    # comments = GPT_prompts.make_prompts(prompt1)
    # my_dict1 = GPT_prompts.make_prompts(prompt2)
    # my_dict2 = GPT_prompts.make_prompts(prompt3)
    # my_prompting.chat_prompts()

    # Outputs
    my_dict1 = {
   "class 1 (non-toxic)": [0, 1, 8, 11, 14, 18, 22, 25, 30, 32, 37, 40, 42, 44, 48, 49],
   "class 2": [6, 9, 13, 16, 17, 20, 24, 28, 33, 36, 38, 39, 41, 43],
   "class 3": [2, 3, 4, 10, 12, 15, 19, 23, 27, 29, 31, 34, 35, 45, 46],
   "class 4": [5, 7, 21, 26],
   "class 5 (fully toxic)": [47]
    }
    my_dict2 = {
   "class 1 (non-toxic)": [52, 55, 57, 58, 63, 64, 68, 71, 73, 76, 79],
   "class 2": [50, 54, 59, 60, 61, 62, 65, 69, 70, 74, 75, 78],
   "class 3": [53, 56, 66, 67, 72, 77],
   "class 4": [51],
   "class 5 (fully toxic)": []
    }

    combined_dict = toxic.combine_dict(df, my_dict1, my_dict2)

    my_classifier = clt()
    # Define my best classifier model
    nb_model = make_pipeline(CountVectorizer(), MultinomialNB())
    model_name = 'Naive Bayes'
    df['comment'] = df['comment'].fillna('')

    X = df['comment']
    y = df['toxicity'][:80]
    X_test = df['comment'][80:]
    y_test = None
    X_train = df['comment'][:80]
    y_train = y
    model_preds = my_classifier.classifier_predictions(nb_model, model_name, X, y, X_test=X_test, y_test=y_test,
                                                       X_train=X_train, y_train=y_train)
    print(model_preds[1])
    df['toxicity'][80:] = model_preds[1]

    df.to_csv('exported_files/crawl.csv')
    df.to_html('exported_files/crawl.html')

    toxic.most_toxic_lang(df)
    pages_set = toxic.highest_toxic_page()
    df['date'] = df['date'].apply(toxic.convert_greek_date)
    df['date'] = pd.to_datetime('now') - df['date']
    toxic.toxic_over_time(df, pages_set)





