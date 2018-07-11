import sys, warnings
warnings.filterwarnings("ignore")
from random import shuffle, sample
import pickle as pk
import gc

import numpy as np
import pandas as pd
import scipy.io
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.recurrent import LSTM
from keras.layers import Merge
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from progressbar import Bar, ETA, Percentage, ProgressBar    
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import spacy
#from spacy.en import English
from src.extract_features import get_questions_matrix_sum, get_images_matrix, get_answers_matrix, get_questions_tensor_timeseries
from src.utils import freq_answers, grouped, get_questions_sum, get_images_matrix, get_answers_sum

training_questions = open("preprocessed/v2/ques_train.txt","rb").read().decode('utf8').splitlines()
training_questions_len = open("preprocessed/v2/ques_train_len.txt","rb").read().decode('utf8').splitlines()
answers_train = open("preprocessed/v2/answer_train.txt","rb").read().decode('utf8').splitlines()
images_train = open("preprocessed/v2/images_coco_id.txt","rb").read().decode('utf8').splitlines()
img_ids = open('preprocessed/v2/coco_vgg_IDMap.txt').read().splitlines()
vgg_path = "data/coco/vgg_feats.mat"

nlp = spacy.load("en_core_web_md")
print ("Loaded WordVec")

vgg_features = scipy.io.loadmat(vgg_path)
img_features = vgg_features['feats']
id_map = dict()
print ("Loaded VGG Weights")

upper_lim = 1000 #Number of most frequently occurring answers in COCOVQA (Covering >80% of the total data)
training_questions, answers_train, images_train = freq_answers(training_questions, 
                                                               answers_train, images_train, upper_lim)
training_questions_len, training_questions, answers_train, images_train = (list(t) for t in zip(*sorted(zip(training_questions_len, 
                                                                                                          training_questions, answers_train, 
                                                                                                          images_train))))
#print (len(training_questions), len(answers_train),len(images_train))

lbl = LabelEncoder()
lbl.fit(answers_train)
nb_classes = len(list(lbl.classes_))
pk.dump(lbl, open('preprocessed/v2/label_encoder_lstm.sav','wb'))

batch_size               =      128
img_dim                  =     4096
word2vec_dim             =      300
#max_len                 =       30 # Required only when using Fixed-Length Padding

num_hidden_nodes_mlp     =     1024
num_hidden_nodes_lstm    =      512
num_layers_mlp           =        3
num_layers_lstm          =        3
dropout                  =       0.5
activation_mlp           =     'tanh'
num_epochs               =         1 
log_interval             =         1 

for ids in img_ids:
    id_split = ids.split()
    id_map[id_split[0]] = int(id_split[1])

image_model = Sequential()
image_model.add(Reshape(input_shape = (img_dim,), target_shape=(img_dim,)))
#image_model.summary()
language_model = Sequential()
language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, 
                        return_sequences=True, input_shape=(None, word2vec_dim)))

for i in range(num_layers_lstm-2):
    language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, return_sequences=True))
language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, return_sequences=False))

model = Sequential()
model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
for i in range(num_layers_mlp):
    model.add(Dense(num_hidden_nodes_mlp, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
model.add(Dense(upper_lim))
model.add(Activation("softmax"))

model_dump = model.to_json()
open('lstm_structure'  + '.json', 'w').write(model_dump)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

for k in range(num_epochs):
    progbar = generic_utils.Progbar(len(training_questions))
    for ques_batch, ans_batch, im_batch in zip(grouped(training_questions, batch_size, 
                                                       fillvalue=training_questions[-1]), 
                                               grouped(answers_train, batch_size, 
                                                       fillvalue=answers_train[-1]), 
                                               grouped(images_train, batch_size, fillvalue=images_train[-1])):
        timestep = len(nlp(ques_batch[-1]))
        X_ques_batch = get_questions_tensor_timeseries(ques_batch, nlp, timestep)
        #print (X_ques_batch.shape)
        X_img_batch = get_images_matrix(im_batch, id_map, img_features)
        Y_batch = get_answers_sum(ans_batch, lbl)
        loss = model.train_on_batch([X_ques_batch, X_img_batch], Y_batch)
        progbar.add(batch_size, values=[('train loss', loss)])
    if k%log_interval == 0:
        model.save_weights("weights/LSTM" + "_epoch_{:02d}.hdf5".format(k))
model.save_weights("weights/MLP" + "_epoch_{:02d}.hdf5".format(k))


