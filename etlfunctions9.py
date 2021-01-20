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

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, hamming_loss
from collections import Counter

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

def get_user_track_ids(token):
    '''
    Parameters:
        - token: client token to connect to Spotify API
        
    Function: Query Spotify API for user-entered Spotify username to get list of track ids for all
    tracks saved to user's Spotify library. Query API 50 tracks at a time for efficiency.
    
    Return: user_track_ids: unique track ids from user's Spotify library
    '''    
    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print('Error in retrieving user track IDs: No token available')
        return False

    limit = 50
    total_tracks = sp.current_user_saved_tracks(limit=1)['total']
    num_queries = math.ceil(total_tracks / limit)

    user_track_ids = []

    for i in range(num_queries):
        results = sp.current_user_saved_tracks(limit=limit, offset=i*limit)
        for item in results['items']:
            user_track_ids.append(item['track']['id'])
    
    return user_track_ids

def get_playlist_track_ids(search_string, num_entries, token) -> str:
    '''
    Parameters:
        - search_string (name of the playlist you want to create)
        - num_entries (number of public playlists in Spotify you want to extract songs from)
        - token: client token to connect to Spotify API
        
    Function: Query Spotify API using search string to return list of playlists. Once
    playlists are obtained, use another API GET request for each playlist to extract its track ids.
    
    Note: Adjusting num_entries has the largest impact on the dataset in terms of sample size. You are
    effectively getting num_entries opinions from the Spotify user base on which songs belong in which
    types of playlist. Higher num_entries = more robust results
    
    Return: unique track ids from all playlists (list format)
    '''

    # Spotify API can only query up to 50 entries at a time, so to get > 50 entries,
    # need to run a new entry that is offset by the number of entries that have already been returned
    limit = 50
    num_queries = math.ceil(num_entries / limit)

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }

    playlist_ids = []
    
    # Iteratively perform keyword search for playlists using Spotify API, increasing offset by 50 each round
    for i in range(num_queries):
        params_pl = [
        ('q', search_string),
        ('type', 'playlist'),
        ('limit', str(limit)),
        ('offset', str(i * limit))
        ]
        try:
            response = requests.get('https://api.spotify.com/v1/search', 
                        headers = headers, params = params_pl, timeout = 5)
            json = response.json()
            for playlist in json['playlists']['items']:
                playlist_ids.append(playlist['id'])
        except:
            print('Playlist search failure')
            return None
    
    # Run a new Spotify API query for each playlist ID to get list of song ID's in that playlist
    track_ids = []
    for playlist_id in playlist_ids:
        try:
            response = requests.get('https://api.spotify.com/v1/playlists/' + playlist_id + '/tracks', 
                        headers = headers, timeout = 5)
            json = response.json()
            for item in json['items']:
                track_ids.append(item['track']['id'])
        except:
            continue
    return list(set(track_ids))

def get_track_artist_album(track_ids, token):
    '''
    Parameters:
        - track_ids: list of Spotify track ids obtained from get_playlist_track_ids
        - token: client token to connect to Spotify API
        
    Function: Spotify API GET request using track id list to get corresponding
    track names and artists
    
    Return: list of dictionaries where each entry contains track artist, track album, and track name
    '''

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }
    
    limit = 50
    num_queries = math.ceil(len(track_ids) / limit)
    tracks_aan = []

    for i in range(num_queries):
        start = i*50
        end = start + 50
        if len(track_ids[start:]) >= 50:
            id50_segment = track_ids[start:end]
        else:
            id50_segment = track_ids[start:]
        
        try:
            id50_segment_query = ','.join(id50_segment)
        except:
            try:
                for track_id in id50_segment:
                    if track_id is None:
                        id50_segment.remove(track_id)
                id50_segment_query = ','.join(id50_segment)
            except:
                continue
   
        params_aan = [('ids', id50_segment_query)]

        try:
            response = requests.get('https://api.spotify.com/v1/tracks/', 
                    headers = headers, params=params_aan, timeout = 5)
            json = response.json()
            index = 0
            for track in json['tracks']:
                tracks_aan.append({'track_id': id50_segment[index], 
                                   'artist': track['album']['artists'][0]['name'], 
                                   'album': track['album']['name'], 
                                   'track': track['name']})
                index += 1
        except:
            print('AAN search failure')
            continue

    return tracks_aan

def get_deezer_album_id(tracks_info, album_id_list, deezer_client_secret):
    '''
    Parameters:
    - tracks_info: List of track dicts, each entry contains track_id, artist, album, and track
    - album_id_list: Empty list for storing album IDs
    - deezer_client_secret: client token needed to connect to Deezer API

    Function: Use get_track_artist_album() to get Spotify track, artist, and album for each track_id obtained through 
    initial get_playlist_track_ids keyword search, then use track, artist, and album as query parameters in 
    the Deezer API to get corresponding Deezer album ID. Append result to album_id_list, which is passed in
    as a parameter and becomes a list of dicts with each entry containing a Spotify track id and a corresponding 
    Deezer album id.

    Return: N/A
    '''

    deezer_token = deezer_client_secret

    search_url = "https://api.deezer.com/search"

    headers = {
    'Accept': 'application/json',
    'response-type': deezer_token
    }

    for track in tracks_info:
        querystring = "artist:\"{0}\" album:\"{1}\" track:\"{2}\"".format(track['artist'], 
                                                                          track['album'], 
                                                                          track['track'])
        params = [('q', querystring)]
        try:
            response = requests.get(search_url, headers=headers, params=params)
            data = response.json()
            album_id_list.append({'spot_track_id': track['track_id'],
                                    'deezer_album_id': str(data['data'][0]['album']['id'])})
        except:
            continue

def get_deezer_album_genres(deezer_album_ids, genres_list, deezer_client_secret):
    '''
    Parameters:
    - deezer_album_ids: List of dicts containing Spotify track ID and corresponding Deezer album ID
    - genres_list: empty list for storing album IDs 
    - deezer_client_secret: client token needed to connect to Deezer API
    
    Function: Use Deezer album ID to get the genre of the album that a given 
    track (indicated by track id) belongs to from the Deezer API. Append to genres_list,
    which is passed in as a parameter and becomes a list of dicts with each entry containing
    a Spotify track id and the genre of the album it belongs to. 
    
    Return: None
    '''
    
    deezer_token = deezer_client_secret
    headers = {
        'Accept': 'application/json',
        'response-type': deezer_token
        }

    for entry in deezer_album_ids:
        album_url = "https://api.deezer.com/album/" + entry['deezer_album_id']
        track_id = entry['spot_track_id']
        try:
            response = requests.get(album_url, headers=headers)
            data = response.json()
            genres_list.append({'trackid': track_id, 
                                'genre': data['genres']['data'][0]['name']})
        except:
            continue

def get_genres(track_ids, token, deezer_client_secret):
    '''
    Parameters:
    - track_ids
    - token: client token needed to connect to Spotify API
    - deezer_client_secret: client token needed to connect to Deezer API

    Function: Get Spotify track information (track name, artist, album) from Spotify track_id, then run
    subfunctions to search for Deezer album id, then corresponding album genre from Deezer
    to identify the genre of a given track

    Return: Pandas dataframe with Spotify track id as key and corresponding track genre as value. To
    be joined with master dataset.
    '''
    
    track_info = get_track_artist_album(track_ids, token)
    album_ids = []
    genres_list = []
    get_deezer_album_id(track_info, album_ids, deezer_client_secret)
    get_deezer_album_genres(album_ids, genres_list, deezer_client_secret)
    
    genres_df = pd.DataFrame(columns=['trackid', 'genre'])
    for genre in genres_list:
        genres_df = genres_df.append(genre, ignore_index=True)
    return genres_df

def get_audio_features(track_ids, token):
    '''
    Parameters:
        - track_ids: list of track ids to get audio features for
        - token: client token to connect to Spotify API
        
    Function: Call Spotify API to get audio features for all track_ids passed in. Pings Spotify API
    for 100 songs at a time (limit) to make queries faster.
    
    Return: Dataframe of track ID's with features. To be joined with master dataset.
    
    Dataframe format: 
    trackid |feature1|feature2|feature3|feature4|...
    --------+--------+--------+--------+--------+...
    1efae1j |   0.4  |    2   |   1.4  |  0.23  |...
    '''
    
    sp = spotipy.Spotify(auth=token)
    try:
        feature_columns = [
            'trackid',
            'danceability', 
            'energy', 
            'loudness', 
            'speechiness', 
            'acousticness', 
            'instrumentalness', 
            'liveness', 
            'valence', 
            'tempo']
        features_df = pd.DataFrame(columns=feature_columns)
        num_iter = math.ceil(len(track_ids)/100)
        i = 0
        for i in range(num_iter):
            start = i*100
            end = start + 100
            # Generate a 100-element long segment of the features_df each iteration
            if len(track_ids[start:]) >= 100:
                try:
                    features_df_segment = sp.audio_features(track_ids[start:end])
                except:
                    continue
            else:
                features_df_segment = sp.audio_features(track_ids[start:])
            
            id_index = i*100 

            for features in features_df_segment:
                try:
                    features_filtered = {key:features[key] for key in feature_columns if key in features}
                    audio_features = {'trackid': track_ids[id_index]}
                    audio_features.update(features_filtered)   
                except:
                    continue
                
                try:
                    features_df = features_df.append(audio_features, ignore_index=True)
                except:
                    print(id_index)
                    
                id_index += 1
            
            i += 1            
        
        return features_df
        
    except:
        print('Error occurred during audio feature extraction')
        return None

def search_and_label(pl_name, num_pl_search, token):
    '''
    Parameters:
        - pl_name: Playlist name user wants to create, passed into keyword search in get_playlist_track_ids()
        - num_pl_search: Max # of playlists to return per keyword search using get_playlist_track_ids()
        - token: client token to connect to Spotify API
        
    Function: Run a Spotify API keyword search for playlists matching the pl_name using get_playlist_track_ids(), 
    then store all of the songs from each playlist by track id. Keeps track of which keyword search 
    the song came from in the labels column.
    
    Return: Dataframe containing class labels. Labels are not mutually exclusive. To be joined with
    master dataset.
    
    Dataframe format: 
    trackid | label
    --------+--------
    1efae1j | hip hop
    12fae31 | vibey
    ...     | ...

    '''
    
    song_labels = pd.DataFrame(columns=['trackid', 'label'])

    queried_track_ids = get_playlist_track_ids(pl_name, num_pl_search, token)
    
    for track_id in queried_track_ids:
        try:
            search_result = {'trackid': track_id, 'label': pl_name}
            song_labels = song_labels.append(search_result, ignore_index=True)
        except:
            continue

    return song_labels

def store_data(pl_name, num_pl_search, token, deezer_client_secret):
    '''
    Parameters:
        - pl_name: Playlist name user wants to create, passed into search_and_label()
        - num_pl_search: Max # of playlists to return per keyword search, passed into search_and_label()
        - token: client token needed to connect to Spotify API
        - deezer_client_secret: client token needed to connect to Deezer API
        
    Function: Get all unique playlist names that are already in data store, and only proceed with function
    if the input pl_name is not already in this list. Call search_and_label() to get track ids and labels, then call
    get_audio_features() on track ids to get features. Finally, call get_genres() to get the Deezer genre for each 
    song. Merge the three dataframe outputs on track_id to get complete dataset, then store to csv file. For future 
    plotly-dash module, this will save to CloudSQL database instead. 
    
    Return: True if csv is written successfully
    
    Dataframe format: 
    trackid |feature1|feature2|feature3|feature4|   label    |     genre     
    --------+--------+--------+--------+--------+------------+-------------
    1e3ae1j |   0.4  |    2   |   1.4  |  0.23  |  90s rock  |  alternative
    '''
    t0 = time.time()
    
    # ----- REPLACE WITH SQL SELECT UNIQUE label FROM [tableName] LOGIC HERE -----
    
    if path.exists('trackdata.csv'):
        existing_data = pd.read_csv('trackdata.csv')
        stored_pls = list(set(existing_data['label'].to_list()))
        
        if pl_name in stored_pls:
            print('{0} already exists in dataset'.format(pl_name))
            return True
            
    # For loop - filter out playlists from pl_names that are already in database
    # If len(filtered_pl_names) > 0, proceed with function
    
    # ----- END ---------------------
    print('Starting data store for new label: {0}'.format(pl_name))
    track_labels = search_and_label(pl_name, num_pl_search, token)
    track_ids = track_labels['trackid'].to_list()
    track_features = get_audio_features(track_ids, token)
    track_genres = get_genres(track_ids, token, deezer_client_secret)

    track_data = pd.concat([track_features, track_labels, track_genres], join='inner', axis=1)
    track_data.reset_index()
    track_data = track_data.loc[:,~track_data.columns.duplicated()]
   
    try:
        # ----- REPLACE WITH SQL INSERT INTO [tableName] LOGIC HERE -----
        
        if path.exists('trackdata.csv'):
            track_data.to_csv('trackdata.csv', mode='a', header=False, index=False)
        else:
            track_data.to_csv('trackdata.csv', mode='w', header=True, index=False)

        # ----- END ---------------------
        
        t1 = time.time()
        print('Success: {0} unique tracks with features and labels obtained in {1} seconds'.format( \
              len(track_data.trackid.unique()), (t1-t0)))

        return True
    
    except:
        print ('Error: CSV write failure')
        return False

def store_user_track_data(token, deezer_client_secret):
    '''
    Parameters:
        - token: client token needed to connect to Spotify API
        - deezer_client_secret: client token needed to connect to Deezer API
        
    Function: Call get_user_track_ids() to get track ids from user's Spotify library, then call
    get_audio_features() on track ids to get features. Finally, call get_genres() to get the Deezer genre for each 
    song. Merge the three dataframe outputs on track_id to get complete dataset, then store to csv file. For future 
    plotly-dash module, this will save to CloudSQL database instead. 
    
    Note: This function is similar to store_data(), but doesn't store labels, as they will be predicted by the model 
    trained on data stored using store_data().
    
    Return: True if csv is written successfully
    
    Dataframe format: 
    trackid |feature1|feature2|feature3|feature4|   genre     
    --------+--------+--------+--------+--------+-------------
    1e3ae1j |   0.4  |    2   |   1.4  |  0.23  | alternative
    '''
    t0 = time.time()

    user_track_ids = get_user_track_ids(token)
    
    # ----- REPLACE WITH SQL SELECT UNIQUE label FROM [tableName] LOGIC HERE -----
    
    if path.exists('user_trackdata.csv'):
        existing_data = pd.read_csv('user_trackdata.csv')
        stored_tracks = list(set(existing_data['trackid'].to_list()))
        track_ids = [track for track in user_track_ids if track not in stored_tracks]
    else:
        track_ids = user_track_ids
            
    # For loop - filter out playlists from pl_names that are already in database
    # If len(filtered_pl_names) > 0, proceed with function
    
    # ----- END ---------------------
    trackids_df = pd.DataFrame({'trackid': track_ids})
    track_features = get_audio_features(track_ids, token)
    track_genres = get_genres(track_ids, token, deezer_client_secret)

    track_data = pd.concat([trackids_df, track_features, track_genres], join='inner', axis=1)
    track_data.reset_index()
    track_data = track_data.loc[:,~track_data.columns.duplicated()]
    
    if len(track_data.index) == 0:
        print('User track data is already up to date with Spotify')
        return True
   
    try:
        # ----- REPLACE WITH SQL INSERT INTO [tableName] LOGIC HERE -----
        
        if path.exists('user_trackdata.csv'):
            track_data.to_csv('user_trackdata.csv', mode='a', header=False, index=False)
        else:
            track_data.to_csv('user_trackdata.csv', mode='w', header=True, index=False)

        # ----- END ---------------------
        
        t1 = time.time()
        print('Success: {0} unique tracks with features stored from user library in {1} seconds'.format( \
              len(track_data.trackid.unique()), (t1-t0)))

        return True
    
    except:
        print ('Error: CSV write failure')
        return False

def get_data(pl_names):
    '''
    Parameters:
        - pl_names: List of labels user wants to include in exploration and model building

    Function: Get data from data store, filtered to the list of playlists desired based on 
    the entries in pl_names. Load output into Pandas dataframe.
    
    Return: Dataframe of track ID's with features and labels filtered by pl_names input.
    
    Dataframe format: 
    trackid |feature1|feature2|feature3|feature4| genre  |      label     
    --------+--------+--------+--------+--------+--------+----------------
    1e3ae1j |   0.4  |    2   |   1.4  |  0.23  |  rock  |  classic rock  
    '''
    
    t0 = time.time()
    
    # --- REPLACE WITH SELECT * FROM [tableName] HERE ---
    
    data = pd.read_csv('trackdata.csv')

    # --- END ---
    
    filtered_data = data[data['label'].isin(pl_names)]
    features = [
            'danceability', 
            'energy', 
            'loudness', 
            'speechiness', 
            'acousticness', 
            'instrumentalness', 
            'liveness', 
            'valence', 
            'tempo']

    t1 = time.time()

    print('{0} records obtained in {1} seconds'.format(len(filtered_data.index), t1-t0))

    return {'data': filtered_data, 'features': features, 'playlists': pl_names}

def get_user_track_data():
    '''
    Parameters:
        - None

    Function: Get user track data from data store and load output into Pandas dataframe.
    
    Return: Dictionary with the following keys:
            data: Dataframe of track ID's with features / genres
            features: list of feature columns
    
    Dataframe format: 
    trackid |feature1|feature2|feature3|feature4|  genre     
    --------+--------+--------+--------+--------+--------
    1e3ae1j |   0.4  |    2   |   1.4  |  0.23  |  rock  
    '''
    
    t0 = time.time()
    
    # --- REPLACE WITH SELECT * FROM [tableName] HERE ---
    
    data = pd.read_csv('user_trackdata.csv')

    # --- END ---
    
    features = [
            'danceability', 
            'energy', 
            'loudness', 
            'speechiness', 
            'acousticness', 
            'instrumentalness', 
            'liveness', 
            'valence', 
            'tempo']

    t1 = time.time()

    print('{0} records obtained in {1} seconds'.format(len(data.index), t1-t0))

    return {'data': data, 'features': features}

def binarize(df, feature_var, label_var, id_col):
    '''
    Parameters:
        - df: dataframe extracted using get_data() 
        - feature_var: categorical feature variable to be binarized
        - label_var: categorical label to be binarized. Can pass 'None' if dataset has no labels
        - id_col: name of unique track id column

    Function: Pass df, feature_var, and label_var as parameters to Pandas get_dummies()
    method to binarize feature_var and label_var into dummy variables. Then use id_col in
    groupby().max() method to account for tracks that belong to more than one feature
    or label category.
    
    Return: Dictionary with the following keys:
            data: Binarized dataframe with new columns for each dummy variable generated
            feature_var: list of new dummy feature columns for backwards translation
            label_var (if applicable): list of new dummy label columns for backwards translation
    
    Dataframe format: 
    trackid |feature1|feature2|feature3|feature4| genre_rock |  genre_pop  | label_country | label_90s jams 
    --------+--------+--------+--------+--------+------------+-------------+---------------+----------------
    1e3ae1j |   0.4  |    2   |   1.4  |  0.23  |     1      |     0       |       0       |        1       
    '''

    feature_var_values = list(set(df[feature_var].to_list()))
    new_feature_var_values = [feature_var + '_' + value for value in feature_var_values]

    if label_var != 'None':
        label_var_values = list(set(df[label_var].to_list()))
        new_label_var_values = [label_var + '_' + value for value in label_var_values]

        binarized_data = pd.get_dummies(data=df, columns=[feature_var, label_var])
        binarized_data = binarized_data.groupby([id_col], as_index=False).max()

        return {'data': binarized_data, feature_var: new_feature_var_values, label_var: new_label_var_values}
    
    else:
        binarized_data = pd.get_dummies(data=df, columns=[feature_var])
        binarized_data = binarized_data.groupby([id_col], as_index=False).max()

        return {'data': binarized_data, feature_var: new_feature_var_values}

def get_user_playlists(token):
    '''
    Parameters:
        - token: client token needed to connect to Spotify API

    Function: Access user's Spotify playlists via Spotify API and return dataframe containing
    playlist names and their corresponding Spotify playlist id's.
    
    Return: Dataframe with each entry containing a playlist name and playlist id.   
    '''
    sp = spotipy.Spotify(auth=token)
    num_playlists = sp.current_user_playlists()['total']
    num_queries = math.ceil(num_playlists / 50)
    user_playlists = pd.DataFrame(columns=['name', 'id'])

    for i in range(num_queries):
        results = sp.current_user_playlists(limit=50)
        for item in results['items']:
           user_playlists = user_playlists.append({'name': item['name'], 'id': item['id']}, ignore_index=True)
    
    return user_playlists

def create_user_playlists(df, token, username):
    '''
    Parameters:
        - df: dataframe containing track id's and predicted playlists they should be entered into
        - token: client token needed to connect to Spotify API
        - username: Spotify username for which playlists are being created

    Function: Create empty playlists for each unique playlist name in df (if they don't already exist), 
    then add them for the specified Spotify user and populate them with the track id's listed in df.
    
    Return: True if playlists were created successfully
    '''
    sp = spotipy.Spotify(auth=token)
    
    try:
        existing_pl_names = get_user_playlists(token)['name'].to_list()
        playlists = list(set(df['playlist'].to_list()))
        playlists_to_add = [pl for pl in playlists if pl not in existing_pl_names]
    except:
        print('Error: Failed to retrieve user playlists')
        return False
    
    for playlist in playlists_to_add:
        try:
            sp.user_playlist_create(username, playlist, public=False, collaborative=False, description='Playlist generated by contextify')
        except:
            print('Error: Failed to create playlist {0}'.format(playlist))
            return False
    
    try:
        user_playlists = get_user_playlists(token)
        generated_playlists = user_playlists[user_playlists['name'].isin(playlists)]
    except:
        print('Error: Failed to retrieve newly added playlists')
        return False
    
    for index, row in generated_playlists.iterrows():       
        try:
            # Add logic to pull track id's from existing playlist, then compare with the list of tracks_to_add and trim tracks_to_add accordingly
            tracks_to_add = df[df['playlist']==row['name']]['trackid'].to_list()
            num_posts = math.ceil(len(tracks_to_add) / 100)

            for i in range(num_posts):
                start = i*100
                end = start + 100
                if len(tracks_to_add[start:]) > 100:
                    sp.user_playlist_add_tracks(username, row['id'], tracks_to_add[start:end])
                else:
                    sp.user_playlist_add_tracks(username, row['id'], tracks_to_add[start:])

        except:
            print('Error: Failed to add songs to playlist {0}'.format(row['name']))
            return False
    
    print('Playlists created successfully')
    return True