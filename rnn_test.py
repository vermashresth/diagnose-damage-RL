from keras.models import load_model
import pickle
import numpy as np
def create_model():

    model = load_model("saved_models/my_modelant4jointsday20_datadiff20timerandpol1.h5")
    print("loading samples")
    pickle_in = open("data_pickles/Ant-v1_4joints20diff5000randpoljamtest1.dict", "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    print(data['class'].shape)
    pred = model.predict(data['bigdata1'])
    print pred[:10]
    a=np.argmax(pred, axis=1)
    print a[:10]
    #print(data['y_data'].reshape(80,6)[::5])
    b=data['class'].flatten()
    print b[:10]
    print np.sum(a==b), len(b)
    print (float(np.sum(a==b))/float(len(b)))

create_model()
