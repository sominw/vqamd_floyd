import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.utils import np_utils

"""Gets the 4096-dimensional CNN features for the given COCO
	VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)
	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)"""

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
	assert not isinstance(img_coco_ids, str)
	nb_samples = len(img_coco_ids)
	nb_dimensions = VGGfeatures.shape[0]
	image_matrix = np.zeros((nb_samples, nb_dimensions))
	for j in range(len(img_coco_ids)):
		image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]

	return image_matrix

"""Sums the word vectors of all the tokens in a question
A numpy array of shape: (nb_samples, word_vec_dim)"""

def get_questions_matrix(questions, nlp):
	assert not isinstance(questions, str)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros((nb_samples, word_vec_dim))
	for i in range(len(questions)):
		tokens = nlp(questions[i])
		for j in range(len(tokens)):
			questions_matrix[i,:] += tokens[j].vector

	return questions_matrix

"""Converts string objects to class labels
	encoder:	a scikit-learn LabelEncoder object
	Output:
	A numpy array of shape (nb_samples, nb_classes)"""

def get_answers_matrix(answers, encoder):
	assert not isinstance(answers, str)
	y = encoder.transform(answers) #string to numerical class
	nb_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, nb_classes)
	return Y

"""Returns a time series of word vectors for tokens in the question
	A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)"""

def get_questions_tensor_timeseries(questions, nlp, timesteps):
	assert not isinstance(questions, str)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
	for i in range(len(questions)):
		tokens = nlp(questions[i])
		for j in range(len(tokens)):
			if j<timesteps:
				questions_tensor[i,j,:] = tokens[j].vector
	return questions_tensor