import operator
from collections import defaultdict
from itertools import zip_longest
import numpy as np
from keras.utils import np_utils

def freq_answers(training_questions, answer_train, images_train, upper_lim):

    freq_ans = defaultdict(int)
    for ans in answer_train:
        freq_ans[ans] += 1

    sort_freq = sorted(freq_ans.items(), key=operator.itemgetter(1), reverse=True)[0:upper_lim]
    top_ans, top_freq = zip(*sort_freq)
    new_answers_train = list()
    new_questions_train = list()
    new_images_train = list()
    for ans, ques, img in zip(answer_train, training_questions, images_train):
        if ans in top_ans:
            new_answers_train.append(ans)
            new_questions_train.append(ques)
            new_images_train.append(img)

    return (new_questions_train, new_answers_train, new_images_train)

def grouped(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_questions_sum(questions, nlp):

    assert not isinstance(questions, str)
    nb_samples = len(questions)
    word2vec_dim = nlp(questions[0])[0].vector.shape[0]
    ques_matrix = np.zeros((nb_samples, word2vec_dim))
    for index in range(len(questions)):
        tokens = nlp(questions[index])
        for j in range(len(tokens)):
            ques_matrix[index,:] += tokens[j].vector

    return ques_matrix

def get_answers_sum(answers, encoder):
    assert not isinstance(answers, str)
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y

def get_images_matrix(img_id, img_map, vgg_features):
    assert not isinstance(img_id,str)
    nb_samples = len(img_id)
    nb_dimensions = vgg_features.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_id)):
        image_matrix[j,:] = vgg_features[:,img_map[img_id[j]]]

    return image_matrix


def most_freq_answer(values):
    ans_dict = {}
    for index in range(10):
        ans_dict[values[index]['answer']] = 1
    for index in range(10):
        ans_dict[values[index]['answer']] += 1

    return max(ans_dict.items(), key = operator.itemgetter(1))[0]
