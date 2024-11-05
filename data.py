import pandas as pd
from sklearn.model_selection import train_test_split


def process_data():
    df = pd.read_csv("data/isarcasm/isarcasm2022.csv")
    df = df[['tweet', 'sarcastic']]
    df.dropna(inplace=True)

    x_train, x_val, y_train, y_val = train_test_split(df['tweet'], df['sarcastic'], test_size=0.2, random_state=34)

    train = pd.DataFrame([x_train, y_train]).T
    train = train.reset_index()[['tweet', 'sarcastic']]
    train.to_csv("data/isarcasm/isarc_train.csv")

    test = pd.DataFrame([x_val, y_val]).T
    test = test.reset_index()[['tweet', 'sarcastic']]
    test.to_csv("data/isarcasm/isarc_test.csv")


def load_data_isarc():
    train = pd.read_csv('data/isarcasm/isarc_train.csv')
    test = pd.read_csv('data/isarcasm/isarc_test.csv')

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()

    return train_tweets, train_labels, test_tweets, test_labels


def load_data_sarc():
    train = pd.read_csv('data/sarc/Train_Dataset.csv')
    test = pd.read_csv('data/sarc/Test_Dataset.csv')

    train_tweets = train['tweet'].values.tolist()
    train_labels = train['sarcastic'].values.tolist()
    test_tweets = test['tweet'].values.tolist()
    test_labels = test['sarcastic'].values.tolist()

    return train_tweets, train_labels, test_tweets, test_labels


def load_data(dataset: str):
    if dataset == "sarc":
        return load_data_sarc()
    elif dataset == "isarc":
        return load_data_isarc()
    print("dataset not found")
    return None