import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datasets import load_dataset

def spacy_tokenizer(document):
    # tokenize the document with spaCY
    doc = nlp(document["texte"])
    # Remove stop words and punctuation symbols
    tokens = [
        token.text for token in doc if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.text.strip() != '' and \
        token.text.find("\n") == -1)]
    return tokens

def dfreq(idf, N):
    return (1+N) / np.exp(idf - 1) - 1



data_files = {"train": "swerick_data_random_train.pkl", "test": "swerick_data_random_test.pkl"}
swerick_dataset = load_dataset("pandas",data_files=data_files)
nlp = spacy.load("sv_core_news_sm", exclude=['parser', 'ner'])
tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda doc :spacy_tokenizer(doc,nlp), 
                                   norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
result = tfidf_vectorizer.fit_transform(swerick_dataset["train"])

