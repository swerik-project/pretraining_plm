from lxml import etree
from pyparlaclarin.read import paragraph_iterator
from pyriksdagen.utils import protocol_iterators
import numpy as np

# We need a parser for reading in XML data
parser = etree.XMLParser(remove_blank_text=True)
import pandas as pd
from sklearn.model_selection import train_test_split

def get_protocols(subset=False):
    if subset ==True :
        return list(protocol_iterators(corpus_root="data/", start=1955, end=1956))
    else:
        return list(protocol_iterators(corpus_root="data/", start=1867, end=202122))
    
def train_test_valid(df,a=0.7,b=0.85):
    df_train=pd.DataFrame()
    df_test=pd.DataFrame(
    )
    df_valid=pd.DataFrame()

    for s in df["protocole"].unique():
        subset_df = df[df["protocole"] == s]
        h=hash(s)
        h_adjusted = h % (2**32 - 1)
        np.random.seed(h_adjusted)
        r = np.random.rand()
        if r <= a :
            df_train = pd.concat([df_train, subset_df], ignore_index=True)
        elif r <=b:
            df_test = pd.concat([df_test, subset_df], ignore_index=True)
        else:
            df_valid = pd.concat([df_valid, subset_df], ignore_index=True)
    return df_train,df_test,df_valid


def create_dataset_swerick(subset=False,train_test_number=0.2): 

    protocols=get_protocols(subset)
    print("Preprocessing of the data...")
    data=[]

    for i in range(len(protocols)):
        protocol_in_question = protocols[i]
        root = etree.parse(protocol_in_question, parser).getroot()
        element_str=""
        for elem in list(paragraph_iterator(root, output="lxml")):
            element_str += " ".join(elem.itertext()).replace("\n","")
            
        data.append({"protocole": i,"texte": "".join(element_str.split())})

    df=pd.DataFrame(data)
    print(df)
    print("Train Test split the dataset...")

    

    df_train,df_test = train_test_split(df,test_size=train_test_number,random_state=42)
    name_train ="swerick_data_train.pkl" if subset else "swerick_data_long_train.pkl"
    name_test ="swerick_data_test.pkl" if subset else "swerick_data_long_test.pkl"
    df_train.to_pickle(name_train)
    df_test.to_pickle(name_test)
    print("done")
    return df_train,df_test


def create_deterministic_swerick_data(subset):
    protocols=get_protocols(subset)

    data=[]

    for i in range(len(protocols)):
        protocol_in_question = protocols[i]
        root = etree.parse(protocol_in_question, parser).getroot()
        element_str=""
        for elem in list(paragraph_iterator(root, output="lxml")):
            element_str += " ".join(elem.itertext()).replace("\n","")
            
        data.append({"protocole": protocol_in_question,"texte": "".join(element_str.split())})

    df=pd.DataFrame(data)
    return train_test_valid(df)