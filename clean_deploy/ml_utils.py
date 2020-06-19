import pandas as pd
import re
import joblib as jb
from scipy.sparse import hstack, csr_matrix
import numpy as np
import json

mapa_meses = {"jan": "Jan",
            "fev": "Feb",
            "mar": "Mar",
            "abr": "Apr",
            "mai": "May",
            "jun": "Jun",
            "jul": "Jul",
            "ago": "Aug",
            "set": "Sep",
            "out": "Oct",
            "nov": "Nov",
            "dez": "Dec"}


mdl_rf = jb.load("random_forest_20200617.pkl.z")
mdl_lgbm = jb.load("lgbm_20200617.pkl.z")
title_vec = jb.load("title_vectorizer_20200617.pkl.z")


def clean_date(data):
    if re.search(r"(\d+) de ([a-z]+)\. de (\d+)", data['watch-time-text']) is None:
        return None

    raw_date_str_list = list(re.search(r"(\d+) de ([a-z]+)\. de (\d+)", data['watch-time-text']).groups())
    # print(raw_date_str_list)
    if len(raw_date_str_list[0]) == 1:
        raw_date_str_list[0] = "0" + raw_date_str_list[0]

    raw_date_str_list[1] = mapa_meses[raw_date_str_list[1]]

    clean_date_str = " ".join(raw_date_str_list)

    return pd.to_datetime(clean_date_str, format="%d %b %Y")

def clean_views(data):
    raw_views_str = re.match(r"(\d+\.?\d*)", data['watch-view-count'])
    if raw_views_str is None:
        return 0
    raw_views_str = raw_views_str.group(1).replace(".", "")
    #print(raw_views_str)

    return int(raw_views_str)

def compute_features(data):

    if 'watch-view-count' not in data:
        return None

    publish_date = clean_date(data)
    if publish_date is None:
        return None


    views = clean_views(data)
    title = data['watch-title']

    features = dict()

    features['publish_time'] = (pd.Timestamp.today() - publish_date) / np.timedelta64(1, 'D')
    features['views'] = views
    features['day_views'] = features['views'] / features['publish_time']
    del features['publish_time']

    vectorized_title = title_vec.transform([title])

    num_features = csr_matrix(np.array([features['views'], features['day_views']]))
    feature_array = hstack([num_features, vectorized_title])

    return feature_array

def compute_prediction(data):
    feature_array = compute_features(data)

    if feature_array is None:
        return 0

    #p_rf = mdl_rf.predict_proba(feature_array)[:,1]
    p_rf = mdl_rf.predict_proba(feature_array)[0][1]
    p_lgbm = mdl_lgbm.predict_proba(feature_array)[0][1]

    p = 0.5*p_rf + 0.5*p_lgbm
    #log_data(data, feature_array, p)

    return p