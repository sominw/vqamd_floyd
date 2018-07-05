import sys
import argparse
import pickle as pk
import warnings
warnings.filterwarnings('ignore')

from progressbar import Bar, ETA, Percentage, ProgressBar    
from keras.models import model_from_json
#from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import spacy

from extract_features import get_questions_matrix_sum, get_images_matrix, get_answers_matrix
from utils import grouped

def main():

    model = model_from_json(open('baseline_mlp.json').read())
    model.load_weights('weights/MLP_epoch_99.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print ("Model Loaded with Weights")

    val_imgs = open('preprocessed/v2/val_images_coco_id.txt','rb').read().decode('utf-8').splitlines()
    val_ques = open('preprocessed/v2/ques_val.txt','rb').read().decode('utf-8').splitlines()
    val_ans = open('preprocessed/v2/answer_val.txt','rb').read().decode('utf-8').splitlines()
    img_ids = open('preprocessed/v2/coco_vgg_IDMap.txt').read().splitlines()
    vgg_path = "data/coco/vgg_feats.mat"

    label_encoder = pk.load(open('preprocessed/v2/label_encoder.sav','rb'))
    vgg_= scipy.io.loadmat(vgg_path)
    vgg_features = vgg_['feats']
    print ("Loaded VGG Features")
    id_map = dict()
    for ids in img_ids:
        id_split = ids.split()
        id_map[id_split[0]] = int(id_split[1])

    print ("Loading en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    n_classes = 1500
    y_pred = []
    batch_size = 128 

    print ("Word2Vec Loaded!")

    widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets)
    #i=1

    for qu_batch,an_batch,im_batch in pbar(zip(grouped(val_ques, batch_size, fillvalue=val_ques[0]), grouped(val_ans, batch_size, fillvalue=val_ans[0]), grouped(val_imgs, batch_size, fillvalue=val_imgs[0]))):
        X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
        X_i_batch = get_images_matrix(im_batch, id_map, vgg_features)
        X_batch = np.hstack((X_q_batch, X_i_batch))
        y_predict = model.predict_classes(X_batch, verbose=0)
        y_pred.extend(label_encoder.inverse_transform(y_predict))
        #print (i,"/",len(val_ques))
        #i+=1
        #print(label_encoder.inverse_transform(y_predict))

    correct_val = 0.0
    total = 0
    f1 = open('res.txt','w')

    for pred, truth, ques, img in zip(y_pred, val_ans, val_ques, val_imgs):
        t_count = 0
        for _truth in truth.split(';'):
            if pred == truth:
                t_count += 1 
        if t_count >=2:
            correct_val +=1
        else:
            correct_val += float(t_count)/3

        total +=1

        try:
            f1.write(str(ques))
            f1.write('\n')
            f1.write(str(img))
            f1.write('\n')
            f1.write(str(pred))
            f1.write('\n')
            f1.write(str(truth))
            f1.write('\n')
            f1.write('\n')
        except:
            pass

    print ("Accuracy: ", correct_val/total)
    f1.write('Final Accuracy is ' + str(correct_val/total))
    f1.close()
    



if __name__ == '__main__':
    main()
