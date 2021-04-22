import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import preprocessing

Data = pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

# Splitting the data
val = pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/SEMI.csv")
x_val = val["text"]


tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, use_idf=True)
tfidf.fit_transform(X)
tfidf.fit_transform(x_val)

X = tfidf.transform(X)
MinMaxScaler = preprocessing.Normalizer()

X = MinMaxScaler.fit_transform(X)
x_val = tfidf.transform(x_val)
x_val = MinMaxScaler.fit_transform(x_val)

svm = SVC(C=10, gamma=1, kernel='rbf', verbose=True)
svm.fit(X, Y)
predictions = svm.predict(x_val)
val["sentiment"] = predictions
val.to_csv("SEMI_PREDICTED.csv")
