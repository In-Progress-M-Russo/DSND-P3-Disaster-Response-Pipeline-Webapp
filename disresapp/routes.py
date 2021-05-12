from disresapp import app
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

import bz2
import _pickle as cPickle
from sqlalchemy import create_engine

import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# load file paths
database_filepath = 'data/DisasterResponses.db'
model_filepath = 'data/cv_trained_model.pkz'

# Load data
# For this we'll use sqlite
engine_dialect = 'sqlite:///'
# full_DB_engine_path = engine_dialect + app.config['db_path']
full_DB_engine_path = engine_dialect + database_filepath

# Assuming Table name
table_name = 'DisasterResponses'
# Create engine with above parameters and load data in a DataFrame
engine = create_engine(full_DB_engine_path)
df = pd.read_sql_table(table_name, engine)

# Load model
data_t = bz2.BZ2File(model_filepath, 'rb')
model_dict = cPickle.load(data_t)
model = model_dict['model']


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract number of messages
    num_messages = df.shape[0]

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract/count the 10 most relevant categories
    cat_vals =  df[df==1][list(df.columns[4:])].sum().sort_values(ascending=False)[:10].values
    cat_names =  df[df==1][list(df.columns[4:])].sum().sort_values(ascending=False)[:10].index.values

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_vals
                )
            ],

            'layout': {
                'title': 'Top 10 Represented Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template(
        'master.html',
        num_messages=num_messages,
        ids=ids,
        graphJSON=graphJSON
    )

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#Custom error route
#Documentation available at https://flask.palletsprojects.com/en/1.1.x/patterns/errorpages/
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
