#### data I/O
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import keras.metrics
from keras import layers
from keras.models import Sequential

print("### READ IN DATA ###")

directory_base_data = os.path.abspath(os.path.realpath(__file__)[:-len(os.path.basename(__file__))] + "../base_data/") + '/'

data = pd.read_json(os.path.join(directory_base_data, 'data.json'))

print("### CREATE TEST CASES ###")

data_sub = data[data['0/1'].apply(lambda x: x == 0 or x == 1)][['0/1', 'title']]

print(data_sub.shape)

data_train, data_test = train_test_split(
    data_sub,
    test_size=500,
    random_state=123,
    stratify=data_sub['0/1']
)

print(data_train.shape)
print(data_test.shape)

data_test, data_val = train_test_split(
    data_test,
    test_size=250,
    random_state=123,
    stratify=data_test['0/1']
)

print(data_test.shape)
print(data_val.shape)

print("We have the following labels and counts:")
print(data_sub.groupby('0/1').count())

data_train, data_val = train_test_split(
    data_train,
    test_size=250,
    random_state=123,
    stratify=data_test['0/1']
)

y_train = data_train['0/1']
y_val = data_val['0/1']
y_test = data_test['0/1']

print("### TOKENIZING ###")

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['title'])

vocab_size = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(data_train['title'].values)
X_val = tokenizer.texts_to_sequences(data_val['title'].values)
X_test = tokenizer.texts_to_sequences(data_test['title'].values)

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

print("### CREATING MODEL ###")

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(
    input_dim=vocab_size, 
    output_dim=embedding_dim, 
    input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=METRICS)
model.summary()