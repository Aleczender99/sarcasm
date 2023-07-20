import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns
from wordcloud import WordCloud

import ml
import denoise

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")

# sarcasm_data = pd.read_csv(r"C:\Users\ciaus\PycharmProjects\licenta\train-balanced-sarcasm.csv")
# data = sarcasm_data['comment']
# labels = sarcasm_data['label']

sarcasm_data = pd.read_json(r"C:\Users\ciaus\PycharmProjects\licenta\Sarcasm_Headlines_Dataset_v2.json", lines=True)
data = sarcasm_data['headline']
labels = sarcasm_data['is_sarcastic']

# non_sarcastic = sarcasm_data[sarcasm_data['label'] == 1]['comment']
# text = ' '.join(str(sentence) for sentence in non_sarcastic)
# image = WordCloud(width=800, height=1200).generate(text)
# plt.imshow(image, interpolation='bilinear')
# plt.axis("off")
# plt.show()

# print(data.str.split().str.len().quantile(0.95))

# lengths = data.str.split().str.len()
# df = pd.DataFrame({'Numarul de cuvinte din propozitie': lengths})
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='Numarul de cuvinte din propozitie', binwidth=1)
# plt.ylabel('Numarul de propozitii')
# plt.xlim(0, 75)
# print('Done')

# Filter out duplicated values
duplicated_data = data.duplicated(keep='first')
data = data[~duplicated_data]
labels = labels[~duplicated_data]

# Filter out null values
null_data = data.isnull()
data = data[~null_data]
labels = labels[~null_data]

data = data.apply(denoise.denoise_data)
print('Finished cleaning')

# Filter out duplicated values
duplicated_data = data.duplicated(keep='first')
data = data[~duplicated_data]
labels = labels[~duplicated_data]

# Filter out null values
null_data = data.isnull()
data = data[~null_data]
labels = labels[~null_data]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.80)

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
tokenizer_json = tokenizer.to_json()
with open('Models/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# max_len = int(np.percentile(data.apply(len), 90))
max_len = 16
vocab_size = len(tokenizer.word_index)

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')



# NB
ml.nb(train_padded, train_labels, test_padded, test_labels)

# CNN
# ml.cnn(train_padded, train_labels, test_padded, test_labels, vocab_size, max_len)

# LTSM
# ml.lstm(train_padded, train_labels, test_padded, test_labels, vocab_size, max_len)

# GRU
# ml.gru(train_padded, train_labels, test_padded, test_labels, vocab_size, max_len)

# Bert
# ml.bert(train_data, train_labels, test_data, test_labels, max_len)


