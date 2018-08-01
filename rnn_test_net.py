from keras.models import load_model
import pickle
def create_model():

    model = load_model("saved_models/my_model.h5")
    print("loading samples")
    pickle_in = open("data_pickles/Hopper-v1_traintest.dict", "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    pred = model.predict(data['bigdata'])
    print(pred)
    print(data['y_data'])

create_model()
