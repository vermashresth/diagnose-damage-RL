from keras.models import load_model
import pickle
import numpy as np
def create_model(data_path, model_path):

    model = load_model(model_path)
    print("loading samples")
    pickle_in = open(data_path, "rb")
    data = pickle.load(pickle_in)
    print("samples loaded")
    print(data['class'].shape)
    pred = model.predict(data['bigdata1'])
    #print pred[:10]
    a=np.argmax(pred, axis=1)
    #print a[:300]
    #print(data['y_data'].reshape(80,6)[::5])
    b=data['class'].flatten()
    #print b[:300]
    for i,j in enumerate(a[a!=b]):
        print j, "-->", b[a!=b][i]
    #print(a[a!=b])
    #print(b[a!=b])
    print np.sum(a==b), len(b)
    print (float(np.sum(a==b))/float(len(b)))

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', type=str)
	parser.add_argument('-m', type=str)

	args = parser.parse_args()

	create_model(args.d, args.m)
