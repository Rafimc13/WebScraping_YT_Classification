import time

import pandas as pd
from selenium.webdriver import Edge
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from Lang_Detector import LangDetect
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from Classification import ClassificationTrain as clt



df = pd.read_csv('exported_files/crawl.csv')



# Outputs
my_dict1 = {
  "class 1 (non-toxic)": [0, 6, 10, 11, 14, 17, 18, 20, 21, 23],
  "class 2": [1, 2, 7, 8, 12, 13, 16, 19, 22, 24],
  "class 3": [3, 9, 15],
  "class 4": [4],
  "class 5 (fully toxic)": [5]
}
my_dict2 = {
    "class 1 (non-toxic)": [25, 26, 27, 28, 29],
    "class 2": [30, 31, 32, 33, 34],
    "class 3": [35, 36, 37, 38, 39],
    "class 4": [40, 41, 42, 43, 44],
    "class 5 (fully toxic)": [45, 46, 47, 48, 49]
}
for key, value in my_dict1.items():
    my_dict1[key] = value + my_dict2[key]

df.loc[my_dict1['class 1 (non-toxic)'], 'toxicity'] = 1
df.loc[my_dict1['class 2'], 'toxicity'] = 2
df.loc[my_dict1['class 3'], 'toxicity'] = 3
df.loc[my_dict1['class 4'], 'toxicity'] = 4
df.loc[my_dict1['class 5 (fully toxic)'], 'toxicity'] = 5


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
model_preds = my_classifier.classifier_predictions(nb_model,model_name, X, y, X_test = X_test, y_test=y_test,
                                                   X_train=X_train, y_train=y_train)

print(model_preds[1])
df['toxicity'][50:] = model_preds[1]
