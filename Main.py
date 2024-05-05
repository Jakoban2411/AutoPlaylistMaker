import sys
import spotipy
import spotipy.util as util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#from sklearn.learning_curve import validation_curve
import statsmodels.api as sm
#from surprise import Reader, Dataset
from sklearn.cluster import KMeans
from scipy.stats import boxcox
from http.server import BaseHTTPRequestHandler, HTTPServer
import re
from urllib.parse import parse_qs
from urllib.parse import urlparse
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import threading

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

def initServer(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print(f'Starting http server on port 8000')
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()
    return httpd

def call_api(username, scope):
    token = util.prompt_for_user_token(username,scope,client_id='5636bde005744293ae571d3dcbac4e7a',client_secret='d13bcaa6e4c04d46a56f5d8530962dde',redirect_uri='http://localhost:8000/') 
    return token

def initSpotipy():
    token = call_api('Jako2411', 'user-top-read')
    sp = spotipy.Spotify(auth=token, requests_session =False, client_credentials_manager =None, oauth_manager =None, auth_manager =None, proxies =None, requests_timeout=30, status_forcelist=429, retries =10, status_retries =10, backoff_factor=10)
    return sp

def all_features(dictionary):
    all_features = []
    for offset in range(0,len(dictionary),100):
        end = offset+100
        if(end>len(dictionary)):
            end=len(dictionary)
        track_ids = [item['track_id'] for item in dictionary[offset:end]]
        features = sp.audio_features(track_ids)
        #analysis = audio_analysis(track_ids)
        
        all_features.extend(features)
        time.sleep(1)
    return all_features

def artist_features(features_for_tracks, dictionary):
    for dic in dictionary:
        for item in features_for_tracks:
            if item == None:
                pass
            else:
                if dic['track_id'] == item['id']:
                    dic['danceability'] = item['danceability']
                    dic['energy'] = item['energy']
                    dic['loudness'] = item['loudness']
                    dic['key'] = item['key']
                    dic['acousticness'] = item['acousticness']
                    dic['valence'] = item['valence']
                    dic['tempo'] = item['tempo']
                    dic['mode'] = item['mode']

def add_features(all_features, dictionary):
    for dic in dictionary:
        for item in all_features:
            if dic['track_id'] == item['id']:
                dic['danceability'] = item['danceability']
                dic['energy'] = item['energy']
                dic['loudness'] = item['loudness']
                dic['key'] = item['key']
                dic['acousticness'] = item['acousticness']
                dic['valence'] = item['valence']
                dic['tempo'] = item['tempo']
                dic['mode'] = item['mode']

def artist_tracks_dict(artist_name):
    list_tracks =[]
    for offset in range(0,500,50):
        x = sp.search(artist_name, 50, offset, type = 'track')
        temp = [track for track in x['tracks']['items']]
        list_tracks.extend(temp)
    top_track_dict = [{'album':item['album']['name'], 'album_id':item['album']['id'], 'album_release':item['album']['release_date'],'artist':item['artists'][0]['name'],'track_name':item['name'],'track_id':item['id']} for item in list_tracks]
    for piece in list_tracks:
        for section in top_track_dict:
            if piece['id'] == section['track_id']:
                if len(piece['artists']) > 1:
                    section['feature'] = piece['artists'][1]['name']
                else:
                    section['feature'] = 'No Feature'
    example = top_track_dict
    feat = all_features(example)
    artist_features(feat,example)
    return top_track_dict

def create_top_tracks_dict(sp, period):
    list_tracks =[]
    for offset in range(0,500,50):
        x = sp.current_user_top_tracks(50,offset,period)
        temp = [track for track in x['items']]
        list_tracks.extend(temp)
    top_track_dict = [{'album':item['album']['name'], 'album_id':item['album']['id'], 'album_release':item['album']['release_date'],'artist':item['artists'][0]['name'],'track_name':item['name'],'track_id':item['id']} for item in list_tracks]
    for item in list_tracks:
        for dic in top_track_dict:
            if item['id'] == dic['track_id']:
                if len(item['artists']) > 1:
                    dic['feature'] = item['artists'][1]['name']
                else:
                    dic['feature'] = 'No Feature'
    return top_track_dict

def audio_analysis(id):
    full_list = []
    analysis = sp.audio_analysis(id)
    segment_inputs = analysis['segments']
    segment_analysis_dict = [{'pitches':round(sum(item['pitches'])/len(item['pitches']),1), 'timbre':round(sum(item['timbre'])/len(item['timbre']),1)} for item in segment_inputs]
    
    pitches_list = []
    timbre_list = []
    # Extract pitches and timbre values
    for segment in segment_analysis_dict:
        pitches_list.append(segment['pitches'])
        timbre_list.append(segment['timbre'])
    # Calculate averages
    avg_pitches = sum(pitches_list) / len(pitches_list)
    avg_timbre = sum(timbre_list) / len(timbre_list)

    full_list.append({'pitches': avg_pitches, 'timbre': avg_timbre})
    return full_list

def create_features(dictionary):
    features = all_features(dictionary)
    return add_features(features, dictionary)

def extend_frame(df):
    df['rank'] = range(1, len(df) + 1)
    df['percentile'] = pd.qcut(1 - df['rank'],20,retbins = False, labels=False)
    return df

def poly_regression(x, y, degree_range, xlim_min, xlim_max, ylim_min, ylim_max, alpha = .75,s = 5,width = 2.5):     #polynomial regression 
    split = train_test_split(x,y)
    X_train, X_test, y_train, y_test = split[0], split[1], split[2],split[3]
    x = X_train
    y = y_train
    for degree in degree_range:
        poly_model = make_pipeline(PolynomialFeatures(degree),
                               RandomForestRegressor()) #change to other regressors
        poly_model.fit(x[:, np.newaxis], y)
        xfit = np.linspace(xlim_min,xlim_max, 10000)
        yfit = poly_model.predict(xfit[:, np.newaxis])

def regression(X_list, target):     #linear regression
    split = train_test_split(X_list, target,random_state=69)
    X_train, X_test, y_train, y_test = split[0], split[1], split[2],split[3]
    X = X_train
    target = y_train
     
    X2 =  sm.add_constant(X)
    est = sm.OLS(target, X2)
    est2 = est.fit()
    return est2.rsquared

def relevant_features(dataframe):
    all_features = ['energy', 'valence', 'tempo','danceability','acousticness','key', 'loudness', 'key', 'mode'] #define your features here
    scaler = MinMaxScaler()
    dataframe['loudness'] = scaler.fit_transform(dataframe[['loudness']]) 
    rs = []
    for feature in all_features:
        yt,max_lambda =boxcox(dataframe.percentile +.01)
        xt,max_lambda=boxcox(dataframe[feature] + .01)
        r = regression(xt, yt)
        rs.append((r,feature))
    return [tuple[1] for tuple in sorted(rs, reverse = True)[0:3]]

def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def all_similarities(user,artist,a_df):
    cos_list = []
    for item in artist:
        x = cos_sim(user,item)
        cos_list.append(x)
    a_df['cos_similarity'] = cos_list

def euclidean_distance(user, artist):   
    return np.sqrt(np.sum((user - artist) ** 2))

def all_distance(user,artist,a_df):
    e_dist = []
    for item in artist:
        x = euclidean_distance(user,item)
        e_dist.append(x)
    a_df['euclidean'] = e_dist

def add_to_playlist(sp, recommendations):
    id_add = list(recommendations['track_id'].to_dict().values())
    token2 = call_api('Jako2411', 'playlist-modify-public')
    sp2 = spotipy.Spotify(auth=token2)
    return_to_dict = recommendations['artist'].to_dict()
    artist_name = list(return_to_dict.values())[0]
    user_id = sp.current_user()["id"]
    sp2.user_playlist_create(user_id,artist_name+ " Auto Playlist", True)
    newest_id = sp2.user_playlists(user_id)['items'][0]['id']
    sp2.user_playlist_add_tracks(user_id, newest_id, id_add, None)


if __name__ == "__main__":

    max_recommendations = 100
    httpd = initServer()

    sp = initSpotipy()

    artist = artist_tracks_dict('Future Islands')
    a_df = pd.DataFrame(artist)
    # a_df = a_df.query("artist == 'The Chemical Brothers'")

    lt = create_top_tracks_dict(sp, 'long_term')
    lt_features = create_features(lt)
    st = create_top_tracks_dict(sp, 'short_term')
    st_features = create_features(st)
    mt = create_top_tracks_dict(sp, 'medium_term')
    mt_features = create_features(mt)
    lt_df = pd.DataFrame(lt)
    st_df = pd.DataFrame(st)
    mt_df = pd.DataFrame(mt)
    lt_df = extend_frame(lt_df)
    mt_df = extend_frame(mt_df)
    st_df = extend_frame(st_df)
    frames = [lt_df, st_df, mt_df]
    all_df = pd.concat(frames)
    all_df.to_csv('Jako.csv')   #Debug csv view
    features = relevant_features(all_df)

    a_df.key = a_df.key/11
    all_df.key = all_df.key/11

    profile = np.array([all_df[feat].mean() for feat in features])
    compare = a_df[[features[0],features[1],features[2]]].values

    all_similarities(profile,compare,a_df)
    a_df.nlargest(max_recommendations,'cos_similarity')

    all_distance(profile,compare,a_df)
    a_df = a_df.drop_duplicates('track_name')
    rec = a_df.nsmallest(max_recommendations,'euclidean')

    add_to_playlist(sp, rec)

    httpd.shutdown()