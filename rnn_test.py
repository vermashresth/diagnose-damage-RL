from keras.models import load_model
import pickle
def create_model():

    model = load_model("saved_models/my_modelant2jointsday6_data3.h5")
    print("loading samples")
    pickle_in = open("data_pickles/Ant-v1_2joints50normal3.dict", "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    pred = model.predict(data['bigdata1'])
    print(pred[::])
    #print(data['y_data'].reshape(80,6)[::5])
    print(data['class'][::])

create_model()
