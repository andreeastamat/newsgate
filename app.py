from flask import Flask, Response
from flask_cors import CORS
import json
import datetime as dt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import pickle
import regex as reg
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv(r'dataset_news(2).csv', sep='/')

df['text'] = df.text.values.astype(str)
df = df[df['title'].map(lambda x: len(str(x)) >= 42)]
df['clean_text'] = df['text']
df = df.drop_duplicates(subset='title', keep='first')

cached_stopwords = stopwords.words('english')


def replace_spec(text):
    regex = r'[^a-zA-z0-9/s]'
    text = reg.sub(regex, ' ', text)
    return text


def process_text(text):
    text = text.lower()
    text = replace_spec(text)
    text_list = str.split(text)
    final_text = []

    for item in text_list:
        if item not in cached_stopwords:
            final_text.append(item)

    return " ".join(final_text)


df['clean_text'] = df['clean_text'].apply(process_text)

svm = pickle.load(open('SVM_Model.pkl', 'rb'))

today = dt.datetime.today()
df['date'] = df.date.fillna(today)
df['date'] = pd.to_datetime(df['date'], utc=True)
date = pd.to_datetime(df['date']).apply(lambda x: x.date())
df['date'] = date

X = df['clean_text']

predicted = svm.predict(X)
prob = svm.predict_proba(X)[:, 1] * 100
df['probability'] = np.around(prob, decimals=2)

df = df.drop('text', 1)
df = df.drop('clean_text', 1)
df = df.drop('author', 1)
# df = df.sort_values(by='probability', ascending=True)
print(df.sample(frac=1))
df = df.sample(frac=1).reset_index(drop=True)

app = Flask(__name__)
CORS(app)


@app.route("/news")
def news():
    return Response(json.dumps(df.to_dict(orient="records"), default=str, indent=4, ensure_ascii=False),
                    mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True)
