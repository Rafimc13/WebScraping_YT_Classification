import openai
import pandas as pd
import os
from dotenv import load_dotenv
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from Classification import ClassificationTrain as clt


class PromptingGPT:

    # Load API key and organization from environment variables
    load_dotenv("secrets.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    ClientOpenAi = openai.OpenAI(
            api_key= openai.api_key,
            organization= openai.organization
        )

    conversation_history = []

    def make_prompts(self, prompt):


        # Combine previous messages with the current prompt
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for msg in self.conversation_history:
            messages.append({'role': 'user', 'content': msg})

        messages.append({'role': 'user', 'content': prompt})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )

        # Extract and print the model's reply
        reply = response.choices[0].message.content
        print(reply)

        # Update conversation history
        self.conversation_history.append(prompt)
        self.conversation_history.append(reply)

    def chat_prompts(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        state = True
        while state:
            pat_close = re.compile(
                r'(Bye|goodnight|ok thank you)', flags=re.IGNORECASE)
            message = input("You: ")
            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                chat_completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
            answer = chat_completion.choices[0].message.content
            print(f"ChatGPT: {answer}")
            messages.append({"role": "assistant", "content": answer})
            if re.search(pat_close, message):
                state = False



GPT_prompts = PromptingGPT()

df = pd.read_csv('exported_files/crawl.csv')
training_comments1 = df['comment'][:25].to_string()
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
           'class 1 (non-toxic): [1,7,13 etc]')+training_comments1


prompt3 = ('Moreover based on the above comments and classification, I would like to classify with the same '
           'pattern from 1 (non-toxic)to 5 (fully toxic) the below sentences. I will give you again an example'
           'For example:'
           'class 1 (non-toxic): [1,7,13 etc]')+training_comments2

# GPT_prompts.make_prompts(prompt1)
# GPT_prompts.make_prompts(prompt2)
# GPT_prompts.make_prompts(prompt3)
# my_prompting.chat_prompts()


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

df.to_csv('exported_files/crawl.csv')
df.to_html('exported_files/crawl.html')
