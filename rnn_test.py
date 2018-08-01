from keras.models import load_model
import pickle
def create_model():

    model = load_model("saved_models/my_modelday2_data3.h5")
    print("loading samples")
    pickle_in = open("data_pickles/Hopper-v1_traintest50.dict", "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    pred = model.predict(data['bigdata3'])
    print(pred)
    print(data['class'])

create_model()
