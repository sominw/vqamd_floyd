import json   #parse json
import spacy  #tokenizing text
import progressbar
import pickle as pk
import scipy.io
from spacy.en import English
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from utils import most_freq_answer, grouped, get_questions_sum
from keras.models import Sequential
import sys
from keras.layers.core import Dense, Dropout, Activation
#def main():

try:
    f1 = open("/home/raunaq/VQAMD/preprocessed/results.txt","w")
    questions_val = open("/home/raunaq/VQAMD/preprocessed/ques_val.txt","r").read()
    answers_val = open("/home/raunaq/VQAMD/preprocessed/answer_val.txt",'r').read()
    image_ids = open("/home/raunaq/VQAMD/preprocessed/images_coco_id_val.txt",'r').read()
    vgg_path = "/home/raunaq/vgg_feats.mat"

    answers_train = open("../preprocessed/answer_train.txt","rb").read().decode('utf8').splitlines()

except IOError:
    print ("Error in opening files")

vgg_features = scipy.io.loadmat(vgg_path)
img_features = vgg_features['feats']
print ("Loaded Vgg features")

try:
    label_encoder = joblib.load("/home/raunaq/VQAMD/preprocessed/label_encoder.sav")
except:
    print ("Label encoder didn't load")

nlp = English()
print ("Loaded word2vec")
weights_path = "MLP_epoch_60.hdf5"

lbl = LabelEncoder()
lbl.fit(answers_train)
nb_classes = len(list(lbl.classes_))
y_predict_text = []
widgets = ['Evaluating ', progressbar.Percentage(), ' ', progressbar.Bar(marker='#',left='[',right=']'),
       ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(widgets=widgets)

#Model Parameters
num_hidden_units = 1024
num_hidden_layers = 3
batch_size = 256
dropout = 0.5
activation = 'tanh'
img_dim = 4096
word2vec_dim = 300
num_epochs = 100
log_interval = 10

# Recreating the model architecture
model = Sequential()
model.add(Dense(num_hidden_units, input_dim=word2vec_dim+img_dim, kernel_initializer='uniform'))
model.add(Dropout(dropout))
for i in range(num_hidden_layers):
    model.add(Dense(num_hidden_units, kernel_initializer='uniform'))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
model.add(Dense(nb_classes, kernel_initializer='uniform'))
model.add(Activation('softmax'))

# Loading weights
model.load_weights(weights_path)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

for qu_batch,an_batch in pbar(zip(grouped(questions_val, batch_size, fillvalue=questions_val[0]),
                                            grouped(answers_val, batch_size, fillvalue=answers_val[0]))):
                                            X_batch = get_questions_sum(qu_batch, nlp)
                                            y_predict = model.predict_classes(X_batch, verbose=0)
                                            y_predict_text.extend(labelencoder.inverse_transform(y_predict))

for prediction, truth, question, image in zip(y_predict_text, answers_val, questions_val):
		temp_count=0
		for _truth in truth.split(';'):
			if prediction == _truth:
				temp_count+=1

		if temp_count>2:
			correct_val+=1
		else:
			correct_val+= float(temp_count)/3

		total+=1
		f1.write(question.encode('utf-8'))
		f1.write('\n')
		f1.write(prediction)
		f1.write('\n')
		f1.write(truth.encode('utf-8'))
		f1.write('\n')

f1.write('Final Accuracy is ' + str(correct_val/total))
f1.close()
f1 = open('../results/overall_results.txt', 'a')
f1.write(args.weights + '\n')
f1.write(str(correct_val/total) + '\n')
f1.close()
print ('Final Accuracy on the validation set is', correct_val/total)
