import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datasets import load_dataset
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import argparse
import csv
import pickle
import preprocessing

from typing import List, Optional, Literal, Union
from huggingface_hub import hf_hub_url, list_repo_files

LanguageOption = Literal[
    "et",
    "pl",
    "sr",
    "ru",
    "sv",
    "no_language_found",
    "ji",
    "hr",
    "el",
    "uk",
    "fr",
    "fi",
    "de",
    "multi_language",
]


def get_files_for_lang_and_years(
    languages: Union[None, List[LanguageOption]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
):
    files = list_repo_files("biglam/europeana_newspapers", repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    parquet_files_filtered_for_lang = [
        f for f in parquet_files if any(lang in f for lang in ["uk", "fr"])
    ]
    filtered_files = [
        f
        for f in parquet_files
        if (min_year is None or min_year <= int(f.split("-")[1].split(".")[0]))
        and (max_year is None or int(f.split("-")[1].split(".")[0]) <= max_year)
    ]
    return [
        hf_hub_url("biglam/europeana_newspapers", f, repo_type="dataset")
        for f in filtered_files
    ]
    
def process_large_text(text, nlp):
    # Divide the text into parts smaller than 1,000,000 characters
    size = 1000000
    parts = [text[i:i+size] for i in range(0, len(text), size)]
    
    # Process each part using spaCy
    processed_parts = [nlp(part) for part in parts]
    
    # You can combine the results or handle them separately depending on your requirements
    return processed_parts


def spacy_tokenizer(document,nlp):
    # tokenize the document with spaCY
    doc = process_large_text(document["texte"], nlp)
    # Remove stop words and punctuation symbols
    tokens = [
        token.text for part in doc for token in part if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.text.strip() != '' and \
        token.text.find("\n") == -1)]
    return tokens

def dfreq(idf, N):
    return (1+N) / np.exp(idf - 1) - 1


def main(args):
    with open("Riksdag.txt", 'r') as file:
        rixvox = [line.strip() for line in file]
    print("government")
    with open("wiki.txt", 'r') as file:
        wiki= [line.strip() for line in file]

    print("wiki")
    with open("newspaper.txt", 'r') as file:
        news = [line.strip() for line in file]
    print("new")
    with open("legal.txt", 'r') as file:
        legal = [line.strip() for line in file]
    print("legal")
   
    with open("social_media_corpus.txt", 'r') as file:
        social= [line.strip() for line in file]
        
    docs= rixvox + news + legal + wiki + social
    docs=docs[:100]
    print(f"Total texts: {len(docs)}")

    nlp = spacy.load("sv_core_news_sm", exclude=['parser', 'ner'])
    print("Tokenizing...")
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda doc :spacy_tokenizer(doc,nlp), 
                                    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    print("TFIDF")
    #docs = swerick_dataset["train"]
    length = len(docs)
    result = tfidf_vectorizer.fit_transform(docs)
    print("Finish")
    idf = tfidf_vectorizer.idf_

    idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
    idf_sorted = idf[idf_sorted_indexes]
    tokens_by_df = np.array(tfidf_vectorizer.get_feature_names_out())[idf_sorted_indexes]
    dfreqs_sorted = dfreq(idf_sorted, length).astype(np.int32)
    tokens_dfreqs = {tok:dfreq for tok, dfreq in zip(tokens_by_df,dfreqs_sorted)}
    tokens_pct_list = [int(round(dfreq/length*100,2)) for token,dfreq in tokens_dfreqs.items()]
    
    with open('tokens_pct_list_KBBERT.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Token', 'Document Frequency Percentage'])
        for token, pct in zip(tokens_dfreqs.keys(), tokens_pct_list):
            writer.writerow([token, pct])
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for the model")
    parser.add_argument("--model_checkpoint", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for checkpoint of the trainer")
    parser.add_argument("--tokenizer", type=str, default="exbert_tokenizer_60k", help="Save location for tokenizer")
    parser.add_argument("--checkpoint_trainer", type=str, default=None, help="Save location for checkpoint of the trainer")
    parser.add_argument("--name", type=str, default="exbert_60k", help="repository name")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    main(args)