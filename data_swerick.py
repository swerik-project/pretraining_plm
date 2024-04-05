from lxml import etree
from pyparlaclarin.read import paragraph_iterator
from pyriksdagen.utils import protocol_iterators

# We need a parser for reading in XML data
parser = etree.XMLParser(remove_blank_text=True)
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataset_swerick(subset=False,train_test_number=0.2):
    if subset ==True :
        protocols = list(protocol_iterators(corpus_root="data/", start=1955, end=1956))
    else:
        protocols = list(protocol_iterators(corpus_root="data/", start=1867, end=202122))

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