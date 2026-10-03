from data_preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report

def train_and_evaluate(path):
    df = load_data(path)
    # df = preprocess_data(df)

    X = df["content"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = joblib.load("models/tfidf_vectorizer.pkl")

    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    print("Model trained")

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))

    return model


def save_model(model, path):
    joblib.dump(model, path)
    print("Model saved")

def main():
    path = 'data/twitter_training_cleaned.csv'
    model_path = "models/nb_model.pkl"

    model = train_and_evaluate(path)

    save_model(model, model_path)

if __name__=="__main__":
    main()
