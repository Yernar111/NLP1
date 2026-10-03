import pandas as pd
import spacy

import re

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def remove_html_regex(text):
    clean = re.sub(r'<(script|style).*?>.*?</\1>', ' ', text, flags=re.DOTALL)
    clean = re.sub(r'<[^>]+>', ' ', clean)
    return re.sub(r'\s+', ' ', clean).strip()

def remove_url(text):
    pattern=re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r' ',text)

def load_data(path):
    return pd.read_csv(path, keep_default_na=False)

def preprocess_text(text):

    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.like_num:
            continue
        lemma = token.lemma_.lower().strip()

        if lemma.isalpha() and len(lemma) > 1:
            tokens.append(lemma)

    return ' '.join(tokens)

def preprocess_data(df):
    df.drop(columns=["2401"], inplace=True)
    df.columns = ['entity', 'sentiment', 'content']
    df.dropna(inplace=True)
    df["content"] = df["content"].apply(remove_html_regex)
    df["content"] = df["content"].apply(remove_url)
    df['content'] = df['content'].apply(preprocess_text)
    df["sentiment"] = df["sentiment"].map({"Positive": 2, "Neutral": 1, "Irrelevant": 1, "Negative": 0})
    df.drop_duplicates(inplace=True)

    df.to_csv('../data/twitter_training_cleaned.csv', index=False)

    return df