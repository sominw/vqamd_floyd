import json   #parse json
import spacy  #tokenizing text
import progressbar
from utils import most_freq_answer

nlp = spacy.load("en")

data = 'training_data'
#Create all relevant data-dumps required
image_set_id = open('../preprocessed/v2/val_images_coco_id.txt','wb')
ann = '../data/v2/v2_mscoco_val2014_annotations.json'
ques = '../data/v2/v2_OpenEnded_mscoco_val2014_questions.json'
ques_compile = open('../preprocessed/v2/ques_val.txt', 'wb')
ques_id = open('../preprocessed/v2/ques_val_id.txt', 'wb')
ques_len = open('../preprocessed/v2/ques_val_len.txt', 'wb')
answer_train = open('../preprocessed/v2/answer_val.txt','wb')

ques = json.load(open(ques,'r'))
questions = ques['questions']
qa = json.load(open(ann,'r'))
annotations = qa['annotations']

progress = progressbar.ProgressBar()
print ("Begin Data Dump...")
for index, q in progress(zip(range(len(questions)),questions)):
    ques_compile.write((q['question'] + '\n').encode('utf8'))
    ques_len.write((str(len(nlp(q['question']))) + '\n').encode('utf8'))
    ques_id.write((str(q['question_id'])+'\n').encode('utf8'))
    image_set_id.write((str(q['image_id'])+'\n').encode('utf8'))
    answer_train.write(most_freq_answer(annotations[index]['answers']).encode('utf8'))
    answer_train.write('\n'.encode('utf8'))

print ("Data dump can be found in ../preprocessed/")
