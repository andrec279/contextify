import spotipy
import spotipy.util as util
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import time
import seaborn as sns
import numpy as np
import itertools
import os
import os.path
from os import path
import yaml
from collections import Counter

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, hamming_loss, silhouette_score
from collections import Counter

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from etlfunctions_dash import (get_user_track_ids, get_playlist_track_ids, get_track_artist_album, 
get_deezer_album_id, get_deezer_album_genres, get_genres, get_audio_features, 
search_and_label, store_data, store_user_track_data, get_data, get_user_track_data, 
binarize, get_user_playlists, create_user_playlists)
from dash_helpers import generate_table, generate_averages_plot, generate_pca_2d_plot, generate_sizes_graph, generate_top_genres

'''===== API Access Management (Needs to be moved into a callback output) ====='''
tokenvars = yaml.load(open('apitokens.yaml'))

# Credentials for Spotify API Client (using my Spotify Developer account)
client_id = tokenvars['spotify_client_id']
client_secret = tokenvars['spotify_client_secret']
scope = tokenvars['spotify_access_scope']
redirect_uri = 'http://localhost:7777/callback'

# Credentials for Deezer API (using my Deezer Developer account)
deezer_client_id = tokenvars['deezer_client_id']
deezer_client_secret = tokenvars['deezer_client_secret']
'''===== API Access Management ====='''


'''===== Initialize App ====='''
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
'''===== Initialize App ====='''


'''===== Load Global Variables ====='''
training_data = get_data('all')
data = training_data['data'].drop('trackid', axis=1)
data_wtrackid = training_data['data']
playlists = training_data['playlists']
features = training_data['features']

playlist_options = []
for playlist in playlists:
    playlist_options.append({'label': playlist, 'value': playlist})

stdscaler = StandardScaler()
'''===== Load Global Variables ====='''

app.layout = html.Div(children=[
    html.H1(
        children='Welcome to Contextify',
        style={'textAlign': 'center'}
    ),

    html.Div(
        children='Rediscover your own music in different contexts. Auto-build Spotify playlists \
                  in just a few clicks.',
        style={'textAlign': 'center'}
    ),

    html.Br(),

    html.Div(
        [
            html.H3(
                children='Step 1: Choose playlists to create',
                style={'textAlign': 'center'}
            ),
            html.Div(
                children='''We'll learn what types of tracks go in these playlists, 
                    then apply our knowledge to your Spotify library.''',
                style={'textAlign': 'center'}
            )
        ]
    ),

    html.Br(),

    html.Div(
        [
            dcc.Dropdown(
                id='playlist_select_dropdown',
                options=playlist_options,
                placeholder='Choose playlists to create',
                multi=True,
                style={'width': '200'}
            ),
            html.Button(id='playlist_select_submit', children='Begin'),
        ],
        style={'width':'100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    ),

    html.Br(),

    html.Div(
        html.H3(
            children='''What we've learned about these playlists''',
            style={'textAlign': 'center'}
        ),
        id='model_data_dashboard_header',
        style={'display': 'none'}
    ),

    dcc.Loading(
        id="graph_loading",
        type="default",
        children=html.Div(id="graph_loading_output")
    ),

    html.Div(
        html.Div(
            [           
                html.Div(
                    [
                        html.H4('Playlist Feature Averages', style={'textAlign': 'center'}),
                        dcc.Graph(id='averages_graph')
                    ],
                    id='averages_graph_container',
                    className='six columns',
                    style={'display': 'none'}
                ),
                
                html.Div(
                    [
                        html.H4('2D (PCA) Playlist Feature Distributions', style={'textAlign': 'center'}),
                        dcc.Graph(id='pca_graph')
                    ],
                    id='pca_graph_container',
                    className='six columns',
                    style={'display': 'none'}
                )    
            ],
            className='row',
            style={'width':'100%', 'display': 'block', 'align-items': 'center', 'justify-content': 'center'}
        ),
    ),

    html.Div(
        html.Div(
            [
                html.Div(
                    [
                        html.H4('Top 3 Playlist Genres', style={'textAlign': 'center'}),
                        html.Div(id='top_genres')
                    ], 
                    id='top_genres_container',
                    className='six columns',
                    style={'display': 'none', 'width':'100%', 'align-items': 'center', 'justify-content': 'center'}
                ),
                
                html.Div(
                    [
                        html.H4('Training Data Sample Sizes', style={'textAlign': 'center'}),
                        dcc.Graph(id='sizes_graph')
                    ], 
                    id='sizes_graph_container',
                    className='six columns',
                    style={'display': 'none'}
                )
            ], 
            className='row',
            style={'width':'100%', 'display': 'block', 'align-items': 'center', 'justify-content': 'center'}
        ),
    ),

    html.Br(),

    html.Div(
        html.Div(
            [
                html.H3('Step 2: Enter your Spotify username', style={'textAlign': 'center'}),
                html.Div('''We'll retrieve your 'Liked' songs and categorize them into the 
                    playlists selected'''),
                html.Br(),
                dcc.Input(id='spotify_username_input', type='text', placeholder='Spotify Username'),
                html.Button(id='spotify_username_submit', children='Load Data'),
                dcc.Loading(
                    id="user_loading",
                    type="default",
                    children=html.Div(id="user_loading_output")
                ),
                html.Br(),
                html.Br()
            ], 
            style={'textAlign': 'center'}
        ),
        style={'display': 'none'},
        id='spotify_username_container'
    ),

    html.Div(
        [
            html.Button(id='run_model_button', children='Create Playlists'),
            html.Br(),
            html.Br()
        ],
        style={'display': 'none'},
        id='run_model_button_container'
    ),

    html.Div(
        html.H3(
            children='''How your music was categorized''',
            style={'textAlign': 'center'}
        ),
        id='user_data_dashboard_header',
        style={'display': 'none'}
    ),

    html.Div(
        html.Div(
            [
                html.Div(
                    children='''Adjust prediction threshold using this slider. Song classifications 
                                will only occur if the probability of assignment is above the selected threshold.'''
                    ),
                html.Br(),
                dcc.Slider(
                    id='threshold_slider',
                    min=0.05,
                    max=0.95,
                    step=0.05,
                    value=0.5,
                    marks={
                        0.1: '0.10',
                        0.2: '0.20',
                        0.3: '0.30',
                        0.4: '0.40',
                        0.5: '0.50',
                        0.6: '0.60',
                        0.7: '0.70',
                        0.8: '0.80',
                        0.9: '0.90'
                    }
                ),
            ],
            style={'display': 'none'},
            id='threshold_slider_container'
        ),
        style={'width':'100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    ),

    dcc.Loading(
        id='run_model_loading',
        type='default',
        children=html.Div(id='run_model_loading_output')
    ),

    html.Div(
        [           
            html.Div(
                [
                    html.H4('Your Playlist Feature Averages', style={'textAlign': 'center'}),
                    dcc.Graph(id='averages_graph_user')
                ],
                id='averages_graph_user_container',
                className='six columns',
                style={'display': 'none'}
            ),
            
            html.Div(
                [
                    html.H4('Your 2D (PCA) Playlist Feature Distributions', style={'textAlign': 'center'}),
                    dcc.Graph(id='pca_graph_user')
                ],
                id='pca_graph_user_container',
                className='six columns',
                style={'display': 'none'}
            )    
        ],
        className='row'
    ),

    html.Div(
        [
            html.Div(
                [
                    html.H4('Your Top 3 Playlist Genres', style={'textAlign': 'center'}),
                    html.Div(id='top_genres_user')
                ], 
                id='top_genres_user_container',
                className='six columns',
                style={'display': 'none'}
            ),
            
            html.Div(
                [
                    html.H4('Your Playlist Sizes', style={'textAlign': 'center'}),
                    dcc.Graph(id='sizes_graph_user')
                ],
                id='sizes_graph_user_container',
                className='six columns',
                style={'display': 'none'}
            )
        ], 
        className='row'
    ),

    html.Div(
        html.Div(
            [
                html.Br(),
                html.Button(id='save_button', children='Save to Spotify'),
                html.Br(),
                dcc.Loading(
                    id='save_loading',
                    type='default',
                    children=html.Div(id='save_loading_output')
                ),
            ],
            style={'display': 'none'},
            id='save_button_container'
        ),
        style={'width':'100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
    ),

    # Intermediate values
    html.Div(id='user_data_predicted', style={'display': 'none'}),
    html.Div(id='user_data_predicted_viz', style={'display': 'none'}),
    html.Div(id='user_data_predicted_std_features', style={'display': 'none'}),
    html.Div(id='model_data', style={'display': 'none'}),
    html.Div(id='model_labels_binzd', style={'display': 'none'}),
    html.Div(id='model_genres_binzd', style={'display': 'none'}),
    html.Div(id='viz_data', style={'display': 'none'}),
    html.Div(id='viz_data_w_id', style={'display': 'none'}),
    html.Div(id='std_features', style={'display': 'none'}),
    html.Div(id='user_data', style={'display': 'none'}),
    html.Div(id='user_data_binzd', style={'display': 'none'}),
    html.Div(id='user_genres_binzd', style={'display': 'none'}),
    html.Div(id='user_data_time', style={'display': 'none'}),
    html.Div(id='user_data_len', style={'display': 'none'}),
])

@app.callback(Output('user_loading_output', 'children'),
              Output('user_data', 'children'),
              Output('user_data_binzd', 'children'),
              Output('user_genres_binzd', 'children'),
              Output('run_model_button_container', 'style'),
              Input('spotify_username_submit', 'n_clicks'),
              State('spotify_username_input', 'value'))
def prep_user_data(n_clicks, username):
    '''Get Spotify username and download their track library data.
    Binarize cat. variables and write results as json to 'user_data' hidden div.'''
    if n_clicks:
        token = util.prompt_for_user_token(
                                        username=username, 
                                        scope=scope, 
                                        client_id=client_id,
                                        client_secret=client_secret,
                                        redirect_uri='http://localhost:7777/callback'
                )
        
        t0 = time.time()
        store_user_track_data(username, token, deezer_client_secret)
        user_tracks_df = get_user_track_data(username)['data']

        binarizer_user = binarize(df=user_tracks_df, feature_var='genre', label_var='None', id_col='trackid')
        binarized_user_data = binarizer_user['data']
        binarized_user_genres = binarizer_user['genre']
        t1 = time.time()

        user_message = '''
            Success: {} tracks retrieved from 'Liked' library for user {} in {:.2f} minutes.
            '''.format(len(user_tracks_df.index), username, (t1-t0)/60)

        return user_message, user_tracks_df.to_json(), binarized_user_data.to_json(),\
               binarized_user_genres, {'display': 'block', 'textAlign': 'center'}
    else:
        return None, {}, {}, {}, {'display': 'none'}

@app.callback(Output('user_data_predicted', 'children'),
              Output('user_data_predicted_viz', 'children'),
              Output('user_data_predicted_std_features', 'children'),
              Output('run_model_loading_output', 'children'),
              Output('user_data_dashboard_header', 'style'),
              Output('threshold_slider_container', 'style'),
              Input('run_model_button', 'n_clicks'),
              Input('model_data', 'children'),
              Input('std_features', 'children'),
              Input('model_labels_binzd', 'children'),
              Input('model_genres_binzd', 'children'),
              Input('user_data_binzd', 'children'),
              Input('user_genres_binzd', 'children'),
              Input('user_data', 'children'),
              Input('threshold_slider', 'value'))
def predict_user_labels(n_clicks, model_data_json, std_features_json, model_labels_binzd, model_genres_binzd,
                        binarized_user_data_json, user_genres_binzd, user_data_json, threshold_slider):
    if n_clicks:
        model_data = pd.read_json(model_data_json)
        user_data_binzd = pd.read_json(binarized_user_data_json)
        user_data = pd.read_json(user_data_json)
        std_features = pd.read_json(std_features_json)

        print('model labels:', model_labels_binzd)

        # Reconcile gaps in binary genre columns between user and training dataset
        model_genres_add = []
        user_genres_add = []
        model_genres_add = [genre for genre in user_genres_binzd if genre not in model_genres_binzd]
        user_genres_add = [genre for genre in model_genres_binzd if genre not in user_genres_binzd]

        for genre in model_genres_add:
            model_data[genre] = 0
        for genre in user_genres_add:
            user_data_binzd[genre] = 0

        bin_user_cols = user_data_binzd.columns.to_list()
        model_data = model_data[bin_user_cols + model_labels_binzd]

        user_genres = [col for col in bin_user_cols if col not in features and col != 'trackid']
        model_genres = user_genres

        # Remainder of user data prep for model prediction
        user_track_ids_col = user_data_binzd['trackid']
        std_user_data = user_data_binzd.drop('trackid', axis=1)
        std_user_data = np.concatenate((stdscaler.fit_transform(std_user_data[features]),
                                        std_user_data[user_genres].to_numpy()), axis=1)

        # Model training
        X = pd.concat([std_features, model_data[model_genres]], join='inner', axis=1, ignore_index=True)
        y = model_data[model_labels_binzd]

        multilogreg = OneVsRestClassifier(LogisticRegression(max_iter=500), n_jobs=-1)
        multilogreg.fit(X, y)

        user_data_probas = pd.DataFrame(multilogreg.predict_proba(std_user_data))
        pl_pred_raw = user_data_probas.applymap(lambda x: 1 if x > threshold_slider else 0)
        pl_pred_raw.columns = model_labels_binzd

        pl_model_output = pd.concat([user_track_ids_col, pl_pred_raw], join='inner', axis=1)
        pl_predictions = pd.DataFrame(columns=['trackid', 'label'])

        for pl in model_labels_binzd:
            pl_category = pl_model_output[pl_model_output[pl]==1]
            for index, row in pl_category.iterrows():
                pl_predictions = pl_predictions.append({'trackid': row['trackid'], 
                                                        'label': pl.replace('label_', '')},
                                                        ignore_index=True)

        print('pl_predictions:', pl_predictions.groupby('label').count())

        user_data_predicted = user_data[['trackid']+features+['genre']].merge(pl_predictions, how='inner', on='trackid')
        user_data_predicted_viz = user_data_predicted.drop('trackid', axis=1).reset_index(drop=True)
        user_data_std_features = pd.DataFrame(stdscaler.fit_transform(user_data_predicted[features]), columns=features)

        return user_data_predicted.to_json(), user_data_predicted_viz.to_json(), user_data_std_features.to_json(), {}, {'display': 'block'},\
               {'display': 'block', 'text-align': 'center', 'align-items': 'center', 'justify-content': 'center', 'width': '60%'}
    else:
        return {}, {}, {}, {}, {'display': 'none'}, {'display': 'none'}

@app.callback(Output('model_data', 'children'),
              Output('viz_data_w_id', 'children'),
              Output('viz_data', 'children'),
              Output('std_features', 'children'),
              Output('model_labels_binzd', 'children'),
              Output('model_genres_binzd', 'children'),
              Output('model_data_dashboard_header', 'style'),
              Input('playlist_select_submit', 'n_clicks'),
              State('playlist_select_dropdown', 'value'))
def prep_training_data(n_clicks, playlists):
    if n_clicks:
        viz_data_w_id = data_wtrackid[data_wtrackid['label'].isin(playlists)].reset_index(drop=True)
        viz_data = data[data['label'].isin(playlists)].reset_index(drop=True)
        std_features = pd.DataFrame(stdscaler.fit_transform(viz_data[features]), columns=features)

        binarizer = binarize(df=viz_data_w_id, feature_var='genre', label_var='label', id_col='trackid')
        model_data = binarizer['data'].reset_index(drop=True)
        model_labels_binzd = binarizer['label']
        model_genres_binzd = binarizer['genre']

        print('prep_training_data labels_binzd:', model_labels_binzd)

        return model_data.to_json(), viz_data_w_id.to_json(), viz_data.to_json(),\
               std_features.to_json(), model_labels_binzd, model_genres_binzd, {'display': 'block'} 
    else: 
        return {}, {}, {}, {}, {}, {}, {'display': 'none'}

@app.callback(Output('sizes_graph', 'figure'),
              Output('sizes_graph_container', 'style'),
              Output('spotify_username_container', 'style'),
              Input('playlist_select_submit', 'n_clicks'),
              Input('viz_data', 'children'),
              Input('playlist_select_dropdown', 'value'))
def show_sizes_graph_training(n_clicks, viz_data_json, playlists):
    if n_clicks:
        viz_data = pd.read_json(viz_data_json)        
        return generate_sizes_graph(viz_data), {'display': 'block'}, {'display': 'block'}
    else:
        return {'data': []}, {'display': 'none'}, {'display': 'none'}

@app.callback(Output('top_genres', 'children'),
              Output('top_genres_container', 'style'),
              Input('playlist_select_submit', 'n_clicks'),
              Input('viz_data', 'children'),
              Input('playlist_select_dropdown', 'value'))
def show_top_genres(n_clicks, viz_data_json, playlists):
    if n_clicks:
        viz_data = pd.read_json(viz_data_json)
        return generate_table(generate_top_genres(viz_data, playlists)), {'display': 'block'}
    else:
        return {}, {'display': 'none'}

@app.callback(Output('pca_graph', 'figure'),
              Output('pca_graph_container', 'style'),
              Output('graph_loading_output', 'children'),
              Output('averages_graph', 'figure'),
              Output('averages_graph_container', 'style'),
              Input('playlist_select_submit', 'n_clicks'),
              Input('playlist_select_dropdown', 'value'),
              Input('viz_data', 'children'),
              Input('std_features', 'children'))
def show_pca_graph_and_avg_graph_training(n_clicks, playlists, viz_data_json, std_features_json):
    if n_clicks:
        viz_data = pd.read_json(viz_data_json)
        std_features = pd.read_json(std_features_json)

        averages_graph = generate_averages_plot(viz_data, std_features, features)
        pca_graph = generate_pca_2d_plot(viz_data, std_features)

        return pca_graph, {'display': 'block'}, {}, averages_graph, {'display': 'block'}
    else:
        return {}, {'display': 'none'}, {}, {}, {'display': 'none'}

@app.callback(Output('pca_graph_user', 'figure'),
              Output('pca_graph_user_container', 'style'),
              Output('averages_graph_user', 'figure'),
              Output('averages_graph_user_container', 'style'),
              Output('save_button_container', 'style'),
              Input('run_model_button', 'n_clicks'),
              Input('playlist_select_dropdown', 'value'),
              Input('user_data_predicted_viz', 'children'),
              Input('user_data_predicted_std_features', 'children'))
def show_pca_graph_and_avg_graph_user(n_clicks, playlists, user_viz_data_json, user_std_features_json):
    if n_clicks:
        user_viz_data = pd.read_json(user_viz_data_json)
        user_std_features = pd.read_json(user_std_features_json)

        averages_graph = generate_averages_plot(user_viz_data, user_std_features, features)
        pca_graph = generate_pca_2d_plot(user_viz_data, user_std_features)

        return pca_graph, {'display': 'block'}, averages_graph, {'display': 'block'}, {'display': 'block'}
    else:
        return {}, {'display': 'none'}, {}, {'display': 'none'}, {'display': 'none'}

@app.callback(Output('sizes_graph_user', 'figure'),
              Output('sizes_graph_user_container', 'style'),
              Input('run_model_button', 'n_clicks'),
              Input('user_data_predicted_viz', 'children'))
def show_sizes_graph_user(n_clicks, user_viz_data_json):
    if n_clicks:
        user_viz_data = pd.read_json(user_viz_data_json)        
        return generate_sizes_graph(user_viz_data), {'display': 'block'}
    else:
        return {'data': []}, {'display': 'none'}

@app.callback(Output('top_genres_user', 'children'),
              Output('top_genres_user_container', 'style'),
              Input('run_model_button', 'n_clicks'),
              Input('user_data_predicted_viz', 'children'),
              Input('playlist_select_dropdown', 'value'))
def show_top_genres_user(n_clicks, user_viz_data_json, playlists):
    if n_clicks:
        user_viz_data = pd.read_json(user_viz_data_json)
        return generate_table(generate_top_genres(user_viz_data, playlists)), {'display': 'block'}
    else:
        return {}, {'display': 'none'}

@app.callback(Output('save_loading_output', 'children'),
              Input('save_button', 'n_clicks'),
              State('spotify_username_input', 'value'),
              State('user_data_predicted', 'children'))
def save_playlists_user(n_clicks, username, user_data_json):
    if n_clicks:
        token = util.prompt_for_user_token(
                                        username=username, 
                                        scope=scope, 
                                        client_id=client_id,
                                        client_secret=client_secret,
                                        redirect_uri='http://localhost:7777/callback'
                )
        
        user_data = pd.read_json(user_data_json)
        user_data = user_data[['trackid', 'label']]
        user_data.columns = ['trackid', 'playlist']

        create_user_playlists(user_data, token, username)

        return {}
    
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)