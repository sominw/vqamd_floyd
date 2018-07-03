import sys
from random import shuffle
import pickle as pk

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import scipy.io
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from sklearn.preprocessing import LabelEncoder
from spacy.en import English
from utils import freq_answers, grouped, get_questions_sum, get_images_matrix, get_answers_sum

def main():

    training_questions = open("/code/preprocessed/ques_train.txt","rb").read().decode('utf8').splitlines()
    answers_train = open("/code/preprocessed/answer_train.txt","rb").read().decode('utf8').splitlines()
    images_train = open("/code/preprocessed/images_coco_id.txt","rb").read().decode('utf8').splitlines()
    img_ids = open('/code/preprocessed/coco_vgg_IDMap.txt').read().splitlines()
    vgg_path = "/input/vgg_feats.mat"

    vgg_features = scipy.io.loadmat(vgg_path)
    img_features = vgg_features['feats']
    id_map = dict()
    nlp = English()
    print ("Loaded WordVec")
    lbl = LabelEncoder()
    lbl.fit(answers_train)
    nb_classes = len(list(lbl.classes_))
    pk.dump(lbl, open('/code/preprocessed/label_encoder.sav','wb'))

    print (vgg_path)
    upper_lim = 1500 #Number of most frequently occurring answers in COCOVQA (85%+)
    training_questions, answers_train, images_train = freq_answers(training_questions, answers_train, images_train, upper_lim)
    print (len(training_questions), len(answers_train),len(images_train))
    num_hidden_units = 1024
    num_hidden_layers = 3
    batch_size = 128
    dropout = 0.5
    activation = 'tanh'
    img_dim = 4096
    word2vec_dim = 300
    num_epochs = 100
    log_interval = 10

    for ids in img_ids:
        id_split = ids.split()
        id_map[id_split[0]] = int(id_split[1])

    model = Sequential()
    model.add(Dense(num_hidden_units, input_dim=word2vec_dim+img_dim, kernel_initializer='uniform'))
    model.add(Dropout(dropout))
    for i in range(num_hidden_layers):
        model.add(Dense(num_hidden_units, kernel_initializer='uniform'))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
    model.add(Dense(nb_classes, kernel_initializer='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    tensorboard = TensorBoard(log_dir='/output/Graph', histogram_freq=0, write_graph=True, write_images=True)

    for k in range(num_epochs):
        index_shuffle = list(range(len(training_questions)))
        shuffle(index_shuffle)
        training_questions = [training_questions[i] for i in index_shuffle]
        answers_train = [answers_train[i] for i in index_shuffle]
        images_train = [images_train[i] for i in index_shuffle]
        progbar = generic_utils.Progbar(len(training_questions))
        for ques_batch, ans_batch, im_batch in zip(grouped(training_questions, batch_size, fillvalue=training_questions[-1]), grouped(answers_train, batch_size, fillvalue=answers_train[-1]), grouped(images_train, batch_size, fillvalue=images_train[-1])):
            X_ques_batch = get_questions_sum(ques_batch, nlp)
            X_img_batch = get_images_matrix(im_batch, id_map, img_features)
            X_batch = np.hstack((X_ques_batch, X_img_batch))
            Y_batch = get_answers_sum(ans_batch, lbl)
            loss = model.train_on_batch(X_batch, Y_batch,callbacks= [tensorboard])
            progbar.add(batch_size, values=[('train loss', loss)])

        if k%log_interval == 0:
            model.save_weights("/output/MLP" + "_epoch_{:02d}.hdf5".format(k))
    model.save_weights("/output/MLP" + "_epoch_{:02d}.hdf5".format(k))

if __name__ == '__main__':
    main()
