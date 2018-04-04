# -*- coding: utf-8 -*-
import io
import os

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def get_word(fname, word, option = False):
    """return rep vector of word according to fasttext """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())

    for line in fin:
        if option:
            token = line.split(',')
            if token[0] == word.decode('utf8'):
                print word
                print 'OK'
                return map(float,token[1:] )


        else:
            tokens = line.rstrip().split(' ')


            if tokens[0]== word.decode('utf8'):
                print word
                print 'OK'
                return map(float, tokens[1:])
    print word
    print 'NOT OK'
    return []
def get_label_list(namefile):
    """ Return the list of labels from namefile"""
    list_label = []
    first = True
    with open(namefile, 'r') as f:
        for line in f:
            if not first:
                new_line = line.split(';')
                qt = int(new_line[1].replace('\n', ''))
                list_label.append(int(qt))
            else:
                first = False
    return list_label

def get_sentences_list(namefile_input):
    list_sent = []
    file = open(namefile_input)
    first =True
    for line in file:
        if not first:
            new_line = line.replace('\n', '')
            sent = new_line.split(';')[1]
            list_sent.append(sent)
        else:
            first = False
    return list_sent

def create_file_for_fasttext(namefile_label, namefile_sent, namefile_output):
    label = get_label_list(namefile_label)
    sentences = get_sentences_list(namefile_sent)
    out = open(namefile_output, 'w')
    for i in range(len(label)):
        out.write(sentences[i] + " __label__ " + str(label[i]) + '\n')
    out.close()

def create_test_file_for_fasttext(namefile_label, namefile_sent, namefile_output):
    label = get_label_list(namefile_label)
    sentences = get_sentences_list(namefile_sent)
    out = open(namefile_output, 'w')
    for i in range(len(label)):
        out.write(sentences[i] + '\n')
    out.close()

def train_model(model_name, sentences, label):
    sentences_file = "temp_file_sent.txt"
    out = open(sentences_file, 'w')
    for i in range(len(label)):
        out.write(sentences[i] + " __label__" + str(label[i]) + '\n')
    out.close()
    bashCommand = "fastText/fasttext supervised -input " + sentences_file +  " -output " + model_name
    os.system(bashCommand)


def get_label(model_name, sentences, nb_ind):
    sentences_file = "fastText/temp_file_sent_test.txt"
    out = open(sentences_file, 'w')
    for i in range(len(sentences)):
        out.write(sentences[i] + '\n')
    out.close()
    print sentences_file
    bashCommand =  "fastText/fasttext predict "  + model_name +".bin " + sentences_file +" " +  str(nb_ind) + ' >' + "temp_file_fasttext.txt"
    print bashCommand
    os.system(bashCommand)
    f = open("temp_file_fasttext.txt", 'r')
    list_labels = [[] for i in range(nb_ind)]
    for line in f:
        new_line = line.replace('\n', '')
        new_line = new_line.split('__label__')
        real_line= [int(new_line[i]) for i in range(len(new_line)) if new_line[i]!=""]
        for j in range(len(real_line)):
            list_labels[j].append(real_line[j])
    return list_labels






#create_file_for_fasttext('data/label.csv', 'data/created/train/input_train_norm_medoc_corrected_v2.csv', 'data/created/train/train_for_fasttext.csv')
#create_test_file_for_fasttext('data/label.csv', 'data/created/train/input_train_norm_medoc_corrected_v2.csv', 'data/created/train/test_train_for_fasttext.csv')