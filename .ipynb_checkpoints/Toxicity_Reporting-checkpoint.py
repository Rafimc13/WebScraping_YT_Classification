import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from Classification import ClassificationTrain as clt
from GPT_Prompting import PromptingGPT


class Generate_Toxicity:

    def combine_dict(self, df, dict1, dict2):
        for key, value in my_dict1.items():
            dict1[key] = value + dict2[key]

        df.loc[dict1['class 1 (non-toxic)'], 'toxicity'] = 1
        df.loc[dict1['class 2'], 'toxicity'] = 2
        df.loc[dict1['class 3'], 'toxicity'] = 3
        df.loc[dict1['class 4'], 'toxicity'] = 4
        df.loc[dict1['class 5 (fully toxic)'], 'toxicity'] = 5

        return dict1

    def most_toxic_lang(self):
        pass

    def highest_toxic_page(self):
        pass

    def toxic_over_time(self):
        pass

    def toxic_increases_over_time(self):
        pass











if __name__ == "__main__":

    GPT_prompts = PromptingGPT()
    toxic = Generate_Toxicity()

    df = pd.read_csv('exported_files/crawl.csv')
    training_comments1 = df['comment'][:50].to_string()
    training_comments2 = df['comment'][25:50].to_string()

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
    comments = GPT_prompts.make_prompts(prompt1)
    my_dict1 = GPT_prompts.make_prompts(prompt2)
    my_dict2 = GPT_prompts.make_prompts(prompt3)
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

    X = df['comment']
    y = df['toxicity'][:50]
    X_test = df['comment'][50:]
    y_test = None
    X_train = df['comment'][:50]
    y_train = y
    model_preds = my_classifier.classifier_predictions(nb_model, model_name, X, y, X_test=X_test, y_test=y_test,
                                                       X_train=X_train, y_train=y_train)

    print(model_preds[1])
    df['toxicity'][50:] = model_preds[1]

    df.to_csv('exported_files/crawl.csv')
    df.to_html('exported_files/crawl.html')

