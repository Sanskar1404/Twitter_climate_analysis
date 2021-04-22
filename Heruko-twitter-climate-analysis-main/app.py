from flask import Flask, render_template, request, redirect, url_for
from get_tweets import get_related_tweets
from nltk.stem import PorterStemmer
import regex as re
from nltk.corpus import stopwords
import pickle
import nltk
nltk.download("stopwords")
import pandas as pd
with open("Tfidf.pkl", 'rb') as data:
    tfidf = pickle.load(data)

with open("model.pkl", 'rb') as data2:
    model = pickle.load(data2)


def requestResults(name):
    tweets = get_related_tweets(name)
    tweets = pd.DataFrame(tweets)
    print(tweets)
    print(tweets.columns)
    corpus = []
    for i in range(len(tweets)):
        # Removing Hashtags
        review = re.sub(r'#', '', tweets['tweet_text'][i])
        # Removing Chines
        review = re.sub(r'[^\x00-\x7F]+', '', tweets['tweet_text'][i])
        # Removing Retweets
        review = re.sub(r'RT[\s]+', '', tweets['tweet_text'][i])
        # Removing HyperLinks
        review = re.sub(r'https?:\/\/\s+', '', tweets['tweet_text'][i])
        # selecting characters only
        review = re.sub('[^a-zA-Z]', ' ', tweets['tweet_text'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    Data = tfidf.transform(corpus)
    tweets['prediction'] = model.predict(Data)
    data = str(tweets.prediction.value_counts()) + '\n\n'
    return tweets["prediction"], tweets["tweet_text"]


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))


@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(requestResults(name)) + " </xmp> "


if __name__ == '__main__':
    app.run()
