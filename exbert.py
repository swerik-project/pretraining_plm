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


def token_list(args):
    data_files = {"train": "swerick_data_random_train.pkl", "test": "swerick_data_random_test.pkl"}
    swerick_dataset = load_dataset("pandas",data_files=data_files)
    nlp = spacy.load("sv_core_news_sm", exclude=['parser', 'ner'])
    print("Tokenizing...")
    tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda doc :spacy_tokenizer(doc,nlp), 
                                    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    print("TFIDF")
    docs = swerick_dataset["train"]
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
    
    with open('tokens_pct_list.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Token', 'Document Frequency Percentage'])
        for token, pct in zip(tokens_dfreqs.keys(), tokens_pct_list):
            writer.writerow([token, pct])
            
            
def main(args):
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    num_epochs= args.epochs
    model_name = args.name
    model_checkpoint = args.model_filename
    model = preprocessing.create_model_MLM(model_checkpoint)
    tokenizer =preprocessing.create_tokenizer(args.tokenizer)
    model.resize_token_embeddings(len(tokenizer)) 

    #data_files = {"train": "swerick_data_random_train.pkl", "test": "swerick_data_random_test.pkl"}
    #swerick_dataset = load_dataset("pandas",data_files=data_files)
    #tokenized_datasets =preprocessing.tokenize_dataset(swerick_dataset,tokenizer)
    #lm_datasets = preprocessing.grouping_dataset(tokenized_datasets,chunk_size)

    with open("lm_dataset_exbert.pkl","rb") as fichier:
        lm_datasets=pickle.load(fichier)
    print(lm_datasets)
    #.select(range(5000))
    data_collator = preprocessing.data_collector_masking(tokenizer,0.15)
    logging_steps = len(lm_datasets["train"]) // batch_size
    
    
    trainer = preprocessing.create_trainer(model,model_name,batch_size,logging_steps,train_dataset=lm_datasets["train"],eval_dataset=lm_datasets["test"],data_collator=data_collator,tokenizer=tokenizer,num_epochs=num_epochs)
    trainer.train(resume_from_checkpoint= args.checkpoint_trainer)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_filename", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for the model")
    parser.add_argument("--model_checkpoint", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for checkpoint of the trainer")
    parser.add_argument("--tokenizer", type=str, default="exbert_tokenizer", help="Save location for tokenizer")
    parser.add_argument("--checkpoint_trainer", type=str, default=None, help="Save location for checkpoint of the trainer")
    parser.add_argument("--name", type=str, default="exbert", help="repository name")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    main(args)
    

