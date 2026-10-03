from data_preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(path):
    df = load_data(path)
    # df = preprocess_data(df)

    X = df['MESSAGE']
    y = df["CATEGORY"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tfidf = TfidfVectorizer(max_features=3000)
    
    # X_train = tfidf.fit_transform(X_train)
    # X_test = tfidf.transform(X_test)
    # model = MultinomialNB(alpha=1.0)
    # model.fit(X_train, y_train)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('model', MultinomialNB(alpha=1.0))
    ])

    pipe.fit(X_train, y_train)

    return pipe, X_test, y_test


def evaluate_model(pipe, X_test, y_test):

    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def save_model(pipeline, path):
    joblib.dump(pipeline, path)
    print("Pipeline saved")

def main():
    path = 'data/Spam_Email_cleaned.csv'

    pipeline, X_test, y_test = train_model(path)

    evaluate_model(pipeline, X_test, y_test)

    save_model(pipeline, 'models/pipeline.pkl')

if __name__=="__main__":
    main()
