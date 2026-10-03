import joblib

def predict(sentiment, tfidf_path, model_path, sentiment_types):
    tfidf = joblib.load(tfidf_path)

    sentiment = tfidf.transform(sentiment)

    model = joblib.load(model_path)
    predictions = model.predict(sentiment)
    j=1
    for i in predictions:
        print(f"Prediction {j}:", sentiment_types[i])
        j+=1

if __name__ == "__main__":
    sentiment = [
        "shit",
        "great",
        "amazing",
        "horrible"
    ]
    sentiment_types = ["Negative", "Neutral", "Positive"]
    tfidf_path = 'models/tfidf_vectorizer.pkl'
    model_path = "models/nb_model.pkl"
    

    predict(sentiment, tfidf_path, model_path, sentiment_types)