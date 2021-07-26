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
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

colors = {
    'background': '#f7f7f9'
}

def generate_table(dataframe, max_rows=10):
    return dash_table.DataTable(        
        columns=[{'name': i, 'id': i} for i in dataframe.columns],
        data=dataframe.to_dict('records')
    )

def generate_sizes_graph(dataframe):
    class_sizes = dataframe.groupby('label', as_index=False)\
                      .agg(count=pd.NamedAgg(column='label', aggfunc='count'))
        
    sizes_plot = go.Figure(
                           data = go.Bar(
                               x=class_sizes['label'],
                               y=class_sizes['count'],
                               width=0.5
                           ),
                           layout=go.Layout(
                               paper_bgcolor=colors['background']
                           )
                        )
    
    return sizes_plot

def generate_top_genres(viz_data, playlists):
    top_genres = pd.DataFrame(columns=['Rank'] + playlists)
    for pl in playlists:
        data_slice = viz_data[viz_data['label']==pl]
        genre_counts = data_slice.groupby(['genre']).count().rename(columns={'danceability': 'count'})\
                                 .sort_values(by='count', ascending=False).reset_index()
        total_tracks = len(data_slice.index)
        for index, row in genre_counts.iterrows():
            genre_counts.at[index, 'percent'] = '{:.0f}'.format(row['count']/total_tracks*100) + '%'
        for index, row in genre_counts.iloc[0:3, :].iterrows():
            top_genres.at[index, 'Rank'] = index + 1
            top_genres.at[index, pl] = row['genre'] + ': ' + row['percent']
    return top_genres

def generate_pca_2d_plot(viz_data, std_features):
    pca = PCA(n_components=2, svd_solver='auto')
    labels = viz_data['label']

    pca_features = pd.DataFrame(pca.fit_transform(std_features))
    pca_data = pd.concat([pca_features, labels], join='inner', axis=1)
    pca_data.columns = ['PC1', 'PC2', 'label']

    pca_graph = px.scatter(pca_data, x='PC1', y='PC2', color='label')

    pca_graph.update_layout(
        paper_bgcolor=colors['background']
    )

    return pca_graph

def generate_averages_plot(viz_data, std_features, features):
    labels = viz_data['label']

    averages_data = pd.concat([std_features, labels], join='inner', axis=1)
    averages_data = averages_data.groupby('label').mean()
    averages_graph = go.Figure()

    mins = []
    maxs = []
    for pl in list(set(labels.to_list())):
        pl_averages = averages_data.loc[pl].values
        mins.append(min(pl_averages))
        maxs.append(max(pl_averages))
        averages_graph.add_trace(go.Scatterpolar(
            r=pl_averages,
            theta=features,
            fill='toself',
            name=pl
        ))

    averages_graph.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[min(mins)-0.2, max(maxs)+0.2]
            )),
        showlegend=True,
        paper_bgcolor=colors['background']
    )

    return averages_graph