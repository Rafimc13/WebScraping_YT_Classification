import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score


# Load the dataset
df = pd.read_csv('exported_files/crawl.csv')

# Preprocess the comments
def preprocess(comment):
    # Remove non-alphanumeric characters
    comment = re.sub(r'[^a-zA-Z0-9]', ' ', comment)
    # Convert to lowercase
    comment = comment.lower()
    # Tokenize the comment
    tokens = nltk.word_tokenize(comment)
    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join the tokens back into a string
    comment = ' '.join(tokens)
    return comment

# Apply preprocessing to the comments
df['comment'] = df['comment'].apply(preprocess)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['toxicity_score'], test_size=0.2, random_state=42)

# Vectorize the comments using TF-IDF
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}, Recall: {:.2f}".format(precision, recall))

# Test the model on a new comment
new_comment = "You're a complete idiot for even suggesting such nonsense."
new_comment = preprocess(new_comment)
new_comment = tfidf.transform([new_comment])
toxicity_score = model.predict(new_comment)[0]
print("Toxicity Score: {}".format(toxicity_score))