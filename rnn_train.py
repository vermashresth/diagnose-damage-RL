from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
import pickle

def lstm_model():
   hidden_units = 512  # may increase/decrease depending on capacity needed
   timesteps = 50
   input_dim = 26
   num_classes = 3    # num of classes for ecg output
   model = Sequential()
   model.add(LSTM(hidden_units, input_shape=(timesteps, input_dim)))
   model.add(Dense(256))
   model.add(Dropout(0.25))
   model.add(Activation('relu'))

   model.add(Dense(64))
   model.add(Dropout(0.25))
   model.add(Activation('relu'))
   # output layer
   # model.add(Dense(2))
   

   model.add(Dense(num_classes))
   model.add(Activation('softmax'))
   adam = Adam(lr=0.0001)
   model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
   # model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
   model.summary()
   return model

def train():
   print("loading samples")
   pickle_in = open("data_pickles/Hopper-v1_traindiversebig50.dict", "rb")
   data = pickle.load(pickle_in)
   print("samples loaded")
   xt = data['bigdata1']  # input_data shape = (num_trials, timesteps, input_dim)
   yt = data['class']  # out_data shape = (num_trials, num_classes)
   # yt = data['y_data']
   from sklearn.preprocessing import OneHotEncoder
   enc = OneHotEncoder()
   yt = enc.fit_transform(yt).toarray()
   print(xt.shape, yt.shape)
   batch_size = 64
   epochs = 2
   model = lstm_model()
   model.fit(xt, yt, epochs=epochs, batch_size=batch_size, shuffle=True)
   model.save('saved_models/my_modelday2_data1.h5')

train()
