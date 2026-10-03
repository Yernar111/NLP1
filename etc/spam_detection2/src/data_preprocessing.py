import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

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

    df['MESSAGE'] = df['MESSAGE'].apply(preprocess_text)

    return df