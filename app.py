import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import flask
from flask import Flask, request, render_template, jsonify

# Building Recommender

dataset = pd.read_csv('anime.csv')
dataset.head()
dataset = dataset[0:15000]

genre = dataset['genre']
title = dataset['title']

for i, t in enumerate(title):
    title[i] = t.lower()

m = dataset['score'].quantile(0.8)
top_anime = dataset.copy().loc[dataset['score'] >= m]
top_anime = top_anime.drop_duplicates(subset='title')

top_anime = top_anime.sort_values('score', ascending=False)

tfidf = TfidfVectorizer(stop_words='english')
top_anime['synopsis'] = top_anime['synopsis'].fillna('')
tfidf_matrix = tfidf.fit_transform(top_anime['synopsis'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(top_anime.index, index=top_anime['title']).drop_duplicates()

top_anime = top_anime.sort_values('score', ascending=False)

def get_recs(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]
    anime_indices = [i[0] for i in sim_scores]

    return top_anime['title'].iloc[anime_indices]

features = ['genre']
for feature in features:
    top_anime[feature] = top_anime[feature].apply(literal_eval)

def create_soup(x):
    return ' '.join(x['genre'])

top_anime['soup'] = top_anime.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(top_anime['soup'])

cos_sim_genre = cosine_similarity(count_matrix, count_matrix)

top_anime = top_anime.reset_index()
indices = pd.Series(top_anime.index, index=top_anime['title'])

# Flask App

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

suggestions = top_anime['title'].tolist()
for i, t in enumerate(suggestions):
    suggestions[i] = {'value':t, 'data':t}

@app.route('/search/<string:box>')
def process(box):
    query = request.args.get('query')
    if box == 'title':
        global suggestions
    return jsonify({"suggestions":suggestions})

ind = 0
predicted_list = np.array(top_anime)

@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        anime_title = request.form.get("title")
        pred = get_recs(str(anime_title).casefold(), cos_sim_genre)
        prediction = np.array(pred)
        global predicted_list
        predicted_list = prediction
    global ind
    return render_template("predict.html", prediction = prediction[ind])

@app.route('/prev', methods=['POST'])
def prev_prediction():
    if request.method == 'POST':
        global ind
        if ind > 0:
            ind -= 1
    global predicted_list
    return render_template("predict.html", prediction = predicted_list[ind])

@app.route('/next', methods=['POST'])
def next_prediction():
    if request.method == 'POST':
        global ind
        ind += 1
    global predicted_list
    return render_template("predict.html", prediction = predicted_list[ind])

@app.route('/back', methods=['POST'])
def back():
    if request.method == 'POST':
        global ind
        ind = 0
    return flask.render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
