import random
import pandas as pd
from Lang_Detector import LangDetect
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ClassificationTrain:

    # Creation of instance of class LangDetect
    lang_det = LangDetect()

    # Creating my dataframe with the gold.csv
    sentences_df = pd.read_csv('exported_files\gold.csv')

    # Insert new sentences for training
    new_sentences = lang_det.read_txt('exported_files\more sentences_for train.txt')
    for i in range(len(new_sentences)):
        new_sentences[i] = lang_det.pattern_search(new_sentences[i])

    my_columns = ['sentence', 'language', 'ground_truth', 'author']
    add_sentences_df = pd.DataFrame(columns=my_columns)
    add_sentences_df['sentence'] = new_sentences

    # Creating a ground truth evaluation dataset
    add_sentences_df['ground_truth'][:30] = "english"
    add_sentences_df['ground_truth'][30:55] = "greek"
    add_sentences_df['ground_truth'][55:75] = "greeklish"
    add_sentences_df['ground_truth'][75:] = "other"
    add_sentences_df['author'] = 'chatGPT'

    combined_sent_df = pd.concat([sentences_df, add_sentences_df])

    X = combined_sent_df['sentence']
    y = combined_sent_df['ground_truth']
    n_set = len(X)
    split_set = int(0.5* n_set)

    test_indices = random.sample(range(n_set), n_set - split_set)

    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    X_train = X.iloc[:split_set]
    y_train = y.iloc[:split_set]

    # Define the models
    nb_model = make_pipeline(CountVectorizer(), MultinomialNB())
    svm_model = make_pipeline(CountVectorizer(), LinearSVC(dual=False))
    rf_model = make_pipeline(CountVectorizer(), RandomForestClassifier())

    nb_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    nb_preds = nb_model.predict(X_test)
    svm_preds = svm_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)


    # Evaluate the models
    print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
    print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))


    comments_df = pd.read_csv('exported_files\crawl.csv')

    X_comments = comments_df['comment']
    X_comments = X_comments.fillna('')
    comment_preds_nb = nb_model.predict(X_comments)

    my_vectorizer = nb_model.named_steps['countvectorizer']
    X_new = my_vectorizer.transform(X_comments)

    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X_new.toarray())

    df_plot = pd.DataFrame(data=X_tsne, columns=['X', 'Y', 'Z'])
    df_plot['Predicted Language'] = comment_preds_nb

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for each language
    for lang, color in zip(df_plot['Predicted Language'].unique(), ['r', 'g', 'b']):
        indices = df_plot['Predicted Language'] == lang
        ax.scatter(df_plot.loc[indices, 'X'], df_plot.loc[indices, 'Y'], df_plot.loc[indices, 'Z'], c=color, label=lang)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
