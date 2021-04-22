import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import missingno
import seaborn as sns
import re
import nltk
from sklearn.svm import SVC
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix,\
    f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn import preprocessing
import pickle
MinMaxScaler = preprocessing.MaxAbsScaler()
dataset = pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/twitter_sentiment_data.csv")
print(dataset.head())

print(dataset.shape)
corpus = []
for i in range(0, 43942):
    # Removing Hashtags
    review = re.sub(r'#', '', dataset['message'][i])
    # Removing Chines
    review = re.sub(r'[^\x00-\x7F]+', '', dataset['message'][i])
    # Removing Retweets
    review = re.sub(r'RT[\s]+', '', dataset['message'][i])
    # Removing HyperLinks
    review = re.sub(r'https?:\/\/\s+', '', dataset['message'][i])
    # selecting characters only
    review = re.sub('[^a-zA-Z]', ' ', dataset['message'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
cv = TfidfVectorizer(max_features=40000, ngram_range=(1, 3))

x = cv.fit_transform(corpus)
x = cv.transform(corpus)
y = dataset.iloc[0:43942, 0].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1103)


X_train = MinMaxScaler.fit_transform(X_train)
X_test = MinMaxScaler.fit_transform(X_test)

param_grid = {'C': [10],
              'kernel': ['rbf']}

svmm = SVC(verbose=True)
svm = GridSearchCV(svmm, param_grid=param_grid, refit=True, n_jobs=7, verbose=True, cv=2)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
print("ACCURACY SCORE:", metrics.accuracy_score(y_test, predictions))
print("::::Confusion Matrix::::")
print(confusion_matrix(y_test, predictions))
print("\n")

print(":::Classification Report:::")
print(classification_report(y_test, predictions, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))
print("\n")

print(pd.crosstab(y_test, predictions, rownames=["Orgnl"], colnames=['Predicted']))

pickle.dump(svm, open("model.pkl", "wb"))

print(svm.best_score_)
print(svm.best_params_)
print(svm.error_score)

class_names = ["-1", "0", "1", "2"]
disp = metrics.plot_confusion_matrix(svm, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues)
plt.show()
with open("cv.pickle", "wb") as output:
    pickle.dump(cv, output)
