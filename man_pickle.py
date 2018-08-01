import pickle

print("loading samples")
pickle_in = open("data_pickles/Hopper-v1_trainclas.dict", "rb")
data = pickle.load(pickle_in)
print("samples loaded")
xt = data['bigdata']  # input_data shape = (num_trials, timesteps, input_dim)
yt = data['class']  # out_data shape = (num_trials, num_classes)
yt = data['y_data']
