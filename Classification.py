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

    @staticmethod
    def classifier_predictions(model, model_name, X, y,):
        """Use any classification (sci-kit) model in order to train it
        and predict for new unknown values. Moreover, by using train data
         we print the accuracy of each model"""
        n_set = len(X)
        split_set = int(0.5 * n_set)
        test_indices = random.sample(range(n_set), n_set - split_set)

        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]
        X_train = X.iloc[:split_set]
        y_train = y.iloc[:split_set]

        # Train our model
        model.fit(X_train, y_train)
        # Test our model
        model_preds = model.predict(X_test)

        # Evaluation of the model based on the test set y
        print(f"{model_name} Accuracy: {accuracy_score(y_test, model_preds)}")
        return model, model_preds

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

    # Define some efficient classifier models
    nb_model = make_pipeline(CountVectorizer(), MultinomialNB())
    svm_model = make_pipeline(CountVectorizer(), LinearSVC(dual=False))
    rf_model = make_pipeline(CountVectorizer(), RandomForestClassifier())

    nb_model, nb_preds = classifier_predictions(nb_model, 'Naive Bayes model', X, y)
    svm_model, svm_preds = classifier_predictions(svm_model, 'Support Vector Machines model', X, y)
    rf_model, rf_preds = classifier_predictions(rf_model, 'Random Forests model', X, y)

    # Open the previous file 'crawl.csv' in order to predict with the best classifier
    comments_df = pd.read_csv('exported_files\crawl.csv')

    X_comments = comments_df['comment']
    X_comments = X_comments.fillna('')
    comment_preds_nb = nb_model.predict(X_comments)

    # Using the vectorizer of best classifier in order to vectorize the comments
    my_vectorizer = nb_model.named_steps['countvectorizer']
    X_new = my_vectorizer.transform(X_comments)

    # Transform the vectorized X dataset into a tsne set
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X_new.toarray())

    # Plot our classes in a 3d projection
    df_plot = pd.DataFrame(data=X_tsne, columns=['X', 'Y', 'Z'])
    df_plot['Predicted Language'] = comment_preds_nb

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for each language
    for lang, color in zip(df_plot['Predicted Language'].unique(), ['r', 'g', 'b']):
        indices = df_plot['Predicted Language'] == lang
        ax.scatter(df_plot.loc[indices, 'X'], df_plot.loc[indices, 'Y'], df_plot.loc[indices, 'Z'], c=color,
                   label=lang)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.view_init(elev=30, azim=-38)

    plt.show()

