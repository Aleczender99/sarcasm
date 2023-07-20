import pandas as pd
import tensorflow as tf
import json

import ml
import denoise

from keras_preprocessing.text import Tokenizer, tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import pickle

import warnings

warnings.filterwarnings("ignore")


def predict(database, models, tweets):
    location = 'Models' + '/' + database + '/' + models
    match models:
        case "BERT.h5":
            from transformers import TFBertModel, BertTokenizer
            model = tf.keras.models.load_model(location, custom_objects={"TFBertModel": TFBertModel})
        case "NB.pickle":
            f = open(location, 'rb')
            model = pickle.load(f)
            f.close()
        case "CNN.h5":
            model = tf.keras.models.load_model(location)
        case "LSTM.h5":
            model = tf.keras.models.load_model(location)
        case "GRU.h5":
            model = tf.keras.models.load_model(location)
        case _:
            print("This isn't one of the known models")

    tweet = []
    for tw in tweets:
        tweet.append(denoise.denoise_data(tw['text']))
    print(tweet)

    if models == "NB.pickle":
        max_len = len(model.feature_count_[1])
    else:
        max_len = model.input_shape[1]

    if models == "BERT.h5":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tweet_ids = ml.encoder(tweet, tokenizer, max_len)
        tweets_padded = tf.convert_to_tensor(tweet_ids)
    else:
        location = 'Models' + '/' + database + '/tokenizer.json'
        with open(location) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        tweets_sequences = tokenizer.texts_to_sequences(tweet)
        tweets_padded = pad_sequences(tweets_sequences, maxlen=max_len, padding='post')

    pred = model.predict(tweets_padded)
    pred = list(zip(pred.flatten(), tweets))
    sorted_pred = sorted(pred, key=lambda x: -x[0])

    return sorted_pred
