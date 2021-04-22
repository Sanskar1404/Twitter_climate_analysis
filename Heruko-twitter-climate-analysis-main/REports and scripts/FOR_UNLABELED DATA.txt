import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


Data = pd.read_csv("Semi-Supervised_SVM/Climate_twitter.csv")
# Data['word_counts'] = Data['message'].str.split().str.len()
# Data["Text Length"] = Data["message"].str.len()
# Data.groupby('sentiment')['word_counts'].mean()

# Exploratory analysis
Data.describe()
print(Data.columns)
Data.head()

# print(Data["message"])
# Data Visualization
# sns.histplot(data=Data, x="sentiment", binwidth=0.4, color='lime')
# sns.histplot(x=Data["sentiment"], y=Data["Text Length"], color='blue', binwidth=0.4)

# Checking for missing values
Data.isna().sum()


# Cleaning the data
def msg_cleaning(msg):
    # Removing @abc12
    msg = re.sub(r'@[A-Za-z0-9]+', '', msg)
    # Removing Hashtags
    msg = re.sub(r'#', '', msg)
    # Removing Chines
    msg = re.sub(r'[^\x00-\x7F]+', '', msg)
    # Removing Retweets
    msg = re.sub(r'RT[\s]+', '', msg)
    msg = re.sub(r'rt[\s]+', '', msg)
    # Removing HyperLinks
    msg = re.sub(r'https?:\/\/\s+', '', msg)
    # Removing numeric values
    msg = re.sub(r'\d+', '', msg)
    msg = re.sub(r'aa[A-Za-z0-9]+', '', msg)
    msg = re.sub(r'zz[A-Za-z0-9]+', '', msg)
    return msg


Data['text'] = Data['text'].apply(msg_cleaning)
Data["text"] = Data["text"].str.lower()
# print(Data["message"])


def identify_tokens(row):
    ide_words = row["text"]
    tokens = word_tokenize(ide_words)

    token_words = [w for w in tokens if w.isalpha()]
    return token_words


Data["text"] = Data.apply(identify_tokens, axis=1)
print(Data['text'])


stemming = PorterStemmer()


def stem_list(row):
    my_list = row["text"]
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)


Data["text"] = Data.apply(stem_list, axis=1)
print(Data["text"])

stops = set(stopwords.words("english"))
stops.update(["aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaaaaaaaaaaaah"])


def remove_stops(row):
    my_list = row["text"]
    meningful_words = [w for w in my_list if not w in stops]
    return(meningful_words)


Data["text"] = Data.apply(remove_stops, axis=1)
print(Data["text"])

Data.to_csv("SEMI.csv")
