from keras.models import load_model
import pickle
def create_model():

    model = load_model("saved_models/my_modelant4jointsday8_datadiff20time.h5")
    print("loading samples")
    pickle_in = open("data_pickles/Ant-v1_4joints20normaldiff3.dict", "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    pred = model.predict(data['bigdata1'])
    print(pred[10:20])
    #print(data['y_data'].reshape(80,6)[::5])
    print(data['class'][10:20])

create_model()
