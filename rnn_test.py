from keras.models import load_model
import pickle
def create_model():

    model = load_model("saved_models/my_model3jointsday2_data1.h5")
    print("loading samples")
    pickle_in = open("data_pickles/Hopper-v1_train3joints50timestenormaltest.dict", "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    pred = model.predict(data['bigdata1'])
    print(pred[::5])
    #print(data['y_data'].reshape(80,6)[::5])
    print(data['class'][::5])

create_model()
