import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import nltk
import re
import re,string,unicodedata
from nltk.corpus import stopwords

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential

import warnings
warnings.filterwarnings("ignore")

tw = pd.read_json(r"C:\Users\ciaus\PycharmProjects\pythonProject\tweets.json", lines=True)
tw.head()
tw.shape
tw.isnull().sum()
tw.describe(include='object')
tw['tweet'].duplicated().sum()
tw = tw.drop(tw[tw['tweet'].duplicated()].index, axis=0)


df = pd.read_json(r"C:\Users\ciaus\PycharmProjects\pythonProject\Sarcasm_Headlines_Dataset.json", lines=True)
df.head()
df.info()
df.shape
df.isnull().sum()
df.describe(include='object')
df['headline'].duplicated().sum()
df = df.drop(df[df['headline'].duplicated()].index, axis=0)

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

#Removing the stopwords from text
def split_into_words(text):
    # split into words by white space
    words = text.split()
    return words

def to_lower_case(words):
    # convert to lower case
    words = [word.lower() for word in words]
    return words

def remove_punctuation(words):
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in words]
    stripped = [re_punc.sub('\t', w) for w in words]
    stripped = [re_punc.sub('\n', w) for w in words]
    return stripped

def keep_alphabetic(words):
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
    return words

def remove_stopwords(words):
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

def to_sentence(words):
    # join words to a sentence
    return ' '.join(words)

# Recomving hashtags
def remove_hashtags(words):
    # filter out hashtags
    words = [w for w in words if not w.startswith('#')]
    return words

#Removing the noisy text
def denoise_text(text):
    words = split_into_words(text)
    words = remove_hashtags(words)
    words = to_lower_case(words)
    words = remove_punctuation(words)
    words = keep_alphabetic(words)
    words = remove_stopwords(words)
    return to_sentence(words)

df['headline']=df['headline'].apply(denoise_text)

labels = (df['is_sarcastic'])
data = (df['headline'])

train_ratio = 0.80

train_size = int(len(labels)*train_ratio)

train_data = data[:train_size]
train_labels = labels[:train_size]

test_data = data[train_size:]
test_labels = labels[train_size:]

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)

vocab_size = len(tokenizer.word_index)

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

maxlen = max([len(i) for i in train_sequences])
train_padded = pad_sequences(train_sequences, maxlen=maxlen,  padding='post')
test_padded = pad_sequences(test_sequences, maxlen=maxlen,  padding='post')

index = 10
print(f'sample headline: {train_sequences[index]}')
print(f'padded sequence: {train_padded[index]} \n')

print(f'Original Sentence:  \n {tokenizer.sequences_to_texts(train_sequences[index:index+1])} \n')

# Print dimensions of padded sequences
print(f'shape of padded sequences: {train_padded.shape}')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, 100, input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_padded, np.array(train_labels), validation_data=(test_padded, np.array(test_labels)), epochs=5, verbose=2)

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("Epochs")
plt.ylabel('accuracy')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])
plt.show()

tweets = tw['tweet']
tweets = tweets.apply(denoise_text)

tweets_sequences = tokenizer.texts_to_sequences(tweets)
tweets_padded = pad_sequences(tweets_sequences, maxlen=maxlen,  padding='post')

pred = model.predict(tweets_padded)

plt.figure()
plt.plot(pred)
plt.show()

for i in range(0, len(pred)-1):
    if pred[i] > 0.95:
        print(pred[i], tw['tweet'].values[i])
