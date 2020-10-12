import spotipy
import spotipy.util as util
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import math

'''Get user token to access Spotify API. Token is refreshed periodically, so it's important 
to call this function each time requests are made.'''

username = '1242062883'
client_id ='e6265a912d9c4be18688eee8093bb4e8'
client_secret = 'fc27bdc4c3654450960bbb60c38b3fd0'
redirect_uri = 'http://localhost:7777/callback'
scope = 'playlist-read-private playlist-modify-private user-read-private'

token = util.prompt_for_user_token(username=username, 
                                   scope=scope, 
                                   client_id=client_id,   
                                   client_secret=client_secret,     
                                   redirect_uri='http://localhost:7777/callback')

def get_song_id(song_title: str, artist: str, token: str) -> str:
    '''Given song title and artist, query Spotify API for unique corresponding song ID, 
    which will be used as lookup key to find audio features for each song from Spotify API'''

    query = 'track:' + song_title + ' artist:' + artist

    headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer ' + token,
    }
    params = [
    ('q', query),
    ('type', 'track'),
    ]
    try:
        response = requests.get('https://api.spotify.com/v1/search', 
                    headers = headers, params = params, timeout = 5)
        json = response.json()
        first_result = json['tracks']['items'][0]
        track_id = first_result['id']
        return track_id
    except:
        return None

def get_playlist_track_ids(search_string: str, num_entries: int, token: str) -> str:
    '''Given search string (name of playlist a user wishes to create), query Spotify API 
    for first 500 playlists matching the search string and extract track and artist titles'''

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
            print('Playlist track search failure')
            continue
    return track_ids

def get_audio_features(track_id: str, token: str) -> dict:
    '''Get audio features for a song via Spotify API using song ID as lookup value'''
    sp = spotipy.Spotify(auth=token)
    try:
        features_raw = sp.audio_features([track_id])[0]
        # Features of interest
        audio_features_keys = [
            'danceability', 
            'energy', 
            'loudness', 
            'speechiness', 
            'acousticness', 
            'instrumentalness', 
            'liveness', 
            'valence', 
            'tempo'
            ]
        audio_features = {key:features_raw[key] for key in audio_features_keys if key in features_raw}
        return audio_features
    except:
        return None

pl_name = 'pump up lift'
song_id_df = pd.DataFrame({'song_id': get_playlist_track_ids(pl_name, 100, token)})
song_id_df.groupby(['song_id']).count()

print(song_id_df.head())
