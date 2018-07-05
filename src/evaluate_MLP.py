import sys
import argparse

from progressbar import Bar, ETA, Percentage, ProgressBar    
from keras.models import model_from_json
#from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from extract_features import get_questions_matrix_sum, get_images_matrix, get_answers_matrix
from utils import grouped

def main():

    model = model_from_json(open('baseline_mlp.json').read())
    model.load_weights('weights/MLP_epoch_99.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print ("Model Loaded with Weights")

if __name__ == '__main__':
    main()
