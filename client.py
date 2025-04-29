# client.py
import flwr as fl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import nltk

stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))  
    text = re.sub(r"[^a-zA-Z0-9\s\-]", "", text)  # Retain hyphens
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)  
    text = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words]
    return " ".join(text).strip()  

def preprocess_data(df):
    # 1. prepare data
    df["clean_subject"] = df["subject"].apply(clean_text)
    df["clean_body"] = df["body"].apply(clean_text)

    # Extract sender domain
    df["sender_domain"] = df["sender"].apply(lambda x: x.split("@")[-1] if pd.notnull(x) else "")
    df['sender_domain'] = df['sender_domain'].str[:-1]

    df["receiver_domain"] = df["receiver"].apply(lambda x: x.split("@")[-1] if pd.notnull(x) else "")
    # df['receiver_domain'] = df['receiver_domain'].str[:-1]
    df['receiver_domain'] = df['receiver_domain'].apply(lambda x: x[:-1] if x.endswith(">") else x)

    # Parse date (handle inconsistent formats)
    df["date"] = df["date"].apply(lambda x: pd.to_datetime(x, errors="coerce",utc = True))
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek 
    df['hour_normalized'] = df['hour'] / 23.0

    df = df.dropna(subset=["label", "clean_subject", "clean_body","receiver","subject","date"])

    # Separate the majority and minority classes
    majority_class = df[df['label'] == 1]
    minority_class = df[df['label'] == 0]

    # Randomly sample from the majority class to match the size of the minority class
    balanced_majority_class = majority_class.sample(len(minority_class), random_state=42)

    # Combine the balanced majority class with the minority class
    df_balanced = pd.concat([balanced_majority_class, minority_class])

    # Shuffle the resulting dataframe
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    df_balanced['label'].value_counts()

    df_balanced = df_balanced.drop(columns=["date", "sender", "receiver", "subject", "body","hour"])
    # combine the subject and body for tfidf
    df_balanced['subjectAndBody'] = df_balanced['clean_subject'] + ' ' + df_balanced['clean_body']

    return df_balanced





class SpamClient(fl.client.NumPyClient):
    def __init__(self, df: pd.DataFrame):
        #preprocess the data
        df_clean = preprocess_data(df)
        vectorizer = TfidfVectorizer()
        X_train, X_test, y_train, y_test = train_test_split(
            df_clean.drop(columns='label'), df_clean['label'], test_size=0.2, random_state=50
        )
        self.X_train_tfidif = vectorizer.fit_transform(X_train['subjectAndBody'])
        self.X_test_tfidf = vectorizer.transform(X_test['subjectAndBody'])
        self.y_train = y_train
        self.y_test = y_test
        self.vectorizer = vectorizer
        self.df_clean = df_clean
        # 2. initialize model
        dummy_X = np.zeros((2, self.X_train_tfidif.shape[1]))
        dummy_y = np.array([0, 1])
        self.model = MultinomialNB()
        self.model.partial_fit(dummy_X, dummy_y, classes=[0, 1])

    def get_parameters(self, config):
        # return θ: (feature_log_prob_, class_log_prior_)
        return [self.model.feature_log_prob_, self.model.class_log_prior_]

    def set_parameters(self, parameters, config):
        self.model.feature_log_prob_, self.model.class_log_prior_ = parameters

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        self.model.partial_fit(self.X_train_tfidif, self.y_train, classes=[0,1])
        return self.get_parameters(config), len(self.y_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss = 1 - self.model.score(self.X_test_tfidf, self.y_test)
        return loss, len(self.y_test), {"accuracy": self.model.score(self.X_test_tfidf, self.y_test)}

# start client (pass in its local DataFrame)
if __name__ == "__main__":
    local_data = pd.read_csv("CEAS_08.csv")
    client = SpamClient(local_data)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    # Create a LIME explainer
    pipeline = make_pipeline(client.vectorizer, client.model)
    explainer = LimeTextExplainer(class_names=["safe", "phising"])

    # Choose a sample text to explain
    sample_idx = 0
    sample_text = client.df_clean['subjectAndBody'].iloc[0]

    # Explain the prediction
    exp = explainer.explain_instance(sample_text, pipeline.predict_proba, num_features=10)
    print(f"\nExplaining prediction for:\n{sample_text}\n")
    exp.show_in_notebook()  