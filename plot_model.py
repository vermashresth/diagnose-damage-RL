from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation


hidden_units = 512  # may increase/decrease depending on capacity needed
timesteps = 20
input_dim = 26
num_classes = 8 
model = Sequential()
model.add(LSTM(hidden_units, input_shape=(timesteps, input_dim)))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Dropout(0.25))
model.add(Activation('relu'))
# output layer
model.add(Dense(2))
model.add(Activation('softmax'))

plot_model(model, to_file='model.png')

model = Sequential()
