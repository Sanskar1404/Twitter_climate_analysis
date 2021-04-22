import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,\
    RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix,\
    f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import pickle
Data = pd.read_csv("C:/Users/Um Ar/PycharmProjects/Internship-2/First_processed.csv")
X = Data["message"]
Y = Data["sentiment"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=1103)

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, use_idf=True)
tfidf.fit_transform(X_train)
tfidf.fit_transform(X_test)

X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

MinMaxScaler = preprocessing.Normalizer()
X_train = MinMaxScaler.fit_transform(X_train)
X_test = MinMaxScaler.fit_transform(X_test)

param_grid = {"kernel": ["rbf"]}

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

# pickle.dump(tfidf, open("Tfidf.pkl", "wb"))
# pickle.dump(svm, open("model.pkl", "wb"))

print(svm.best_score_)
print(svm.best_params_)
print(svm.error_score)

class_names = ["-1", "0", "1", "2"]
disp = metrics.plot_confusion_matrix(svm, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
plt.show()


