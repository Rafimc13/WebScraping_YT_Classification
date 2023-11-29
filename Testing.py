import time
import re


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import date2num
import requests
from selenium.webdriver import Edge, EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import WebDriverException
from Lang_Detector import LangDetect
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from Classification import ClassificationTrain as clt
from dateutil import parser



df = pd.read_csv('exported_files/crawl.csv')

print(df)
df.to_csv('exported_files/crawl.csv')
df.to_html('exported_files/crawl.html')
pages_set = set()
for link in df.loc[:, 'link']:
     pages_set.add(link)


def convert_greek_date(greek_date):
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

# Lets convert Dates to Datetime Objects. ChatGPT lines of code
df['date'] = df['date'].apply(convert_greek_date)
df['date'] = pd.to_datetime('now') - df['date']

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
    toxicities = df.loc[(df['link'] == page) & (~df['date'].isna()), 'toxicity'].tolist()  # ChatGPT line of code
    plt.subplot(220+i)
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







