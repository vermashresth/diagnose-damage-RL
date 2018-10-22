import gym

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

X_train = np.random.uniform()
