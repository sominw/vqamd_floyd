import os, argparse, warnings, io, gc
import subprocess
warnings.filterwarnings('ignore')

#warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import flask

import cv2 
import spacy
import numpy as np
import en_core_web_sm
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.set_image_data_format('channels_first')

from models.VQA.VQA import VQA_MODEL
from models.CNN.VGG import VGG_16

app = flask.Flask(__name__)

VQA_weights_file_name   = '/input/models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = '/input/models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = '/input/models/CNN/vgg16_weights.h5'

image_model = VGG_16(CNN_weights_file_name)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
print ("Image model loaded!")
vqa_model = VQA_MODEL()
vqa_model.load_weights(VQA_weights_file_name)
vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print ("VQA Model loaded!")

def get_image_features(image_file_name):

    image_features = np.zeros((1, 4096))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    #im = cv2.resize(image_file_name, (224, 224))
    mean_pixel = [103.939, 116.779, 123.68]
    im = im.astype(np.float32, copy=False)
    for c in range(3):
    	im[:, :, c] = im[:, :, c] - mean_pixel[c]
    im = im.transpose((2,0,1)) # convert the image to RGBA
    im = np.expand_dims(im, axis=0)
    #print (im)
    image_features[0,:] = image_model.predict(im)[0]
    return image_features

def get_question_features(question):

    word_embeddings = en_core_web_sm.load()
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


@app.route("/predict", methods=["POST"])
def predict():
	
	data = {"success": False}

	ques = flask.request.form['ques']
	print (str(ques))
	if flask.request.method == "POST":
		if flask.request.files["image"]:
			#image = flask.request.files["image"].read()
			file = flask.request.files['image']
			#image = cv2.imdecode(np.fromstring(flask.request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
			#print ("File......", file)
			print("\nCrushing Image into Pixels..")
			image_features = get_image_features(str(file.filename))

			print("\nBreaking Words into Vectors..")
			question_features = get_question_features(str(ques))

			print("\nStirring them together with a lil bit of algebra..")

			print("\nPredicting results..\n")
			y_output = vqa_model.predict([question_features, image_features])
			y_sort_index = np.argsort(y_output)
			labelencoder = joblib.load(label_encoder_file_name)
			data["predictions"] = []
			for label in reversed(y_sort_index[0,-5:]):
				r = {"confidence ": str(round(y_output[0,label]*100,2)).zfill(5)+"% ", "label ": labelencoder.inverse_transform(label)}
				data["predictions"].append(r)
				print (str(round(y_output[0,label]*100,2)).zfill(5), "% ", labelencoder.inverse_transform(label))
				data["success"] = True
	gc.collect()

	return flask.jsonify(data)


if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.debug = True
	app.run(host='0.0.0.0')
