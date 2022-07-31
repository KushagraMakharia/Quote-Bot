import pickle
import random
import pandas as pd
from os.path import exists
from gensim.models import FastText
from sklearn.cluster import KMeans


def train():
    file = pd.read_csv("data/quotes.csv")
    file = file[~file.quote.isna()]
    file.author = file.author.fillna(value="Unknown")
    file = create_train_data(file)
    if not (exists('artifacts/embeddings.pkl')):
        embeddings = create_embeddings(file)
    else:
        embeddings = pickle.load(open("artifacts/embeddings.pkl", 'rb'))
    if not (exists('artifacts/model_train_data.pkl')):
        X = []
        for i in range(len(file)):
            if i%1000 == 0:
                print("{} of {}".format(i, len(file)))
            X.append(embeddings[file.loc[i].train_quote])
        pickle.dump(X, open('artifacts/model_train_data.pkl', 'wb'))
    else:
        X = pickle.load(open("artifacts/model_train_data.pkl", 'rb'))
    print("Training data created.")
    model = KMeans(n_clusters=10000, random_state=0)
    model.fit(X)
    file.to_csv("data/reference.csv")
    print("Model trained.")
    filename = 'artifacts/model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print("Score: {}".format(model.score(X)))

def create_embeddings(file):
    vocabulary = []
    for i in range(len(file)):
        if i%1000 == 0:
            print("{} of {}".format(i, len(file)))
        vocabulary.append(file.loc[i].train_quote.split())
    embeddings = FastText(vocabulary)
    filename = 'artifacts/embeddings.pkl'
    pickle.dump(embeddings.wv, open(filename, 'wb'))
    print("Embeddings created.")
    return embeddings.wv


def create_train_data(df):
    df["train_quote"] = df["quote"].str.lower()
    df = df.drop_duplicates(subset=["train_quote"])
    df.reset_index(drop=True, inplace=True)
    df.train_quote = df.train_quote.str.replace('[^a-zA-Z0-9 \n]', ' ', regex=True)
    print("Initial data cleaned.")
    return df

def get_quotes(text):    
    embeddings = pickle.load(open("artifacts/embeddings.pkl", 'rb'))
    model = pickle.load(open("artifacts/model.pkl", 'rb'))
    ref = pd.read_csv("data/reference.csv")
    X = embeddings[text]
    predicted = model.predict([X])
    labels = model.labels_
    index = [i for i, x in enumerate(labels) if x == predicted[0]]
    response = ref.iloc[index].copy()
    response["response"] = response["quote"] + " -" + response["author"]
    return response["response"].tolist()

def return_quote(text):
    quotes = get_quotes(text)
    n = random.randint(0, len(quotes))
    return quotes[n]
