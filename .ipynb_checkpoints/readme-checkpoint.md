1)Creating a Language Detector in order to separate:
a)Greek
b)Greeklish
c)English
d)Other Languages
2) Train my language detector with random sentences that exported from ChatGPT and have stored in a file called 'gold.csv'
3) Crawl many videos on Youtube and take only 
the comments from videos that have Greek/Greeklish titles.
4) Store the comments in a .csv with the name 'crawl'
5) In module classification traing some sci-kit models with the random sentences 'gold.csv' (added more in order to have an efficient dataset)
6) Classification of the true dataset 'crawl.csv' with the comments
7) Plotted the classes in a 3D projection with the help of a vectorizer and see how my classes are separated
8) Module GPT_Prompting in order to make prompts via GPT API (gpt-3.5-turbo-1106)
9) In Toxixity_Reporting ask GPT API to train a prompting classifier and train it with random comments in Greek
10) With this classifier run for a small sample of my real dataset comments in order to store the Ground truth in a dictionary
11) Train the best sci-kit classifier (from previous classification) with the small sample. Classify the rest comments of my dataset
12) Using some functions report: (a) the most toxic language, (b) the page with the more/highest rate
of toxic posts, (c) the page where toxicity is uniform over time, (d) the page where toxicity
increases over time.
