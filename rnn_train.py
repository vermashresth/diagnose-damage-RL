from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
import pickle

def lstm_model(input_shape):
   hidden_units = 512  # may increase/decrease depending on capacity needed
   timesteps = 50
   input_dim = input_shape
   num_classes = 4    # num of classes for ecg output
   model = Sequential()
   model.add(LSTM(hidden_units, input_shape=(timesteps, input_dim)))
   model.add(Dense(256))
   model.add(Dropout(0.25))
   model.add(Activation('relu'))

   model.add(Dense(64))
   model.add(Dropout(0.25))
   model.add(Activation('relu'))

   model.add(Dense(32))
   model.add(Dropout(0.25))
   model.add(Activation('relu'))
   # output layer
   # model.add(Dense(2))
   

   model.add(Dense(num_classes))
   model.add(Activation('softmax'))
   adam = Adam(lr=0.001)
   model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
   # model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
   model.summary()
   return model

def train():
   print("loading samples")
   pickle_in = open("data_pickles/Ant-v1_2joints50normal2000.dict", "rb")
   data = pickle.load(pickle_in)
   print("samples loaded")
   xt = data['bigdata3']  # input_data shape = (num_trials, timesteps, input_dim)
   # yt = data['y_data'].reshape(24000,6)  # out_data shape = (num_trials, num_classes)
   print xt.shape[2]
   yt = data['class']
   from sklearn.preprocessing import OneHotEncoder, normalize
   #xt = normalize(xt)
   enc = OneHotEncoder()
   yt = enc.fit_transform(yt).toarray()
   print(yt[:10])
   print(xt.shape, yt.shape)
   batch_size = 64
   epochs = 300
   model = lstm_model(xt.shape[2])
   model.fit(xt, yt, epochs=epochs, batch_size=batch_size, shuffle=True)
   model.save('saved_models/my_modelant2jointsday6_data3.h5')

train()
