#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
from fast_test_use import load_vectors, get_word
from vocabulary import get_vocabulary_list
punctuation = ['"', '.', ',', ';', ':', '/', '!', '?', '(', ')', '#', '*', '<', '>', '[', ']', "'", '€', '-', "’", "=", '+', '&', '`']
punctuation_selected = ['€', '-', '=', '+', '?', '%']
nb = ['1','2','3','4','5','6','7','8','9', '@card@']
import numpy as np



def create_vectors_from_voc(input_file, voc_file, out_file):
    """ Create the vectors that represent sentences in input_file from voc_file (count nb of time each word in voc) + medoc + punctuation"""
    voc = get_vocabulary_list(voc_file)

    number_line = 0
    with open(input_file, 'r') as f:
        first = True
        for line in f:
            number_line +=1
            print number_line
            if not first:
                out = open(out_file, 'a')
                new_line = line.replace('\n', '')
                new_line = new_line.split(';')
                ind = new_line[0]
                all_sent = new_line[1]
                sent = new_line[1].split(' ')
                size_sent = float(len(sent))
                c = Counter(sent)
                vect = []

                # first numbers appearance
                nb_appear = 0
                for n in nb:
                    if n in all_sent:
                        nb_appear += 1
                vect.append(float(nb_appear))

                # punctuation selected appearance
                for punct in punctuation_selected:
                    if punct in c.keys():
                        vect.append(float(c[punct]))
                    else:
                        vect.append(0.)
                # medocs
                # to deal with

                # nb medoc
                nbhash = all_sent.count('#')
                vect.append(float(nbhash)/2.)

                # vocabulary
                for wrd in voc:
                    if wrd in c.keys():
                        vect.append(float(c[wrd]))
                    else:
                        vect.append(0.)
                vect = np.asarray(vect)/size_sent

                # writing
                out.write(ind )
                print ind
                for n in vect:

                    out.write(',' + str(n))
                out.write('\n')
                out.close()
            else:
                first = False

def create_mean_vectors_fromfasttext(namefile, fastext_file, out_file):
    dict_voc_already_searched = {}
    dict_word_to_ignore = []
    number_line = 0
    out = open(out_file, 'a')
    start = False
    with open(namefile, 'r') as f:
        first = True
        for line in f:
            number_line += 1
            print number_line
            if not first:
                new_line = line.replace('\n', '')
                new_line = new_line.split(';')
                ind = new_line[0]
                if int(ind) >= 9354:
                    all_sent = new_line[1]
                    sent = new_line[1].split(' ')
                    size_sent = float(len(sent))
                    c = Counter(sent)
                    vect = []

                    # first numbers appearance
                    nb_appear = 0
                    for n in nb:
                        if n in all_sent:
                            nb_appear += 1
                    vect.append(float(nb_appear)/float(size_sent))

                    # nb medoc
                    nbhash = all_sent.count('#')
                    vect.append(float(nbhash) / 4. / float(size_sent))

                    # vocabulary
                    mean_vect_voc = np.zeros([1, 300]) # 300 is the size of vector in fasttext
                    nb_word_counted = 0.

                    for wrd in sent:
                        if len(wrd) >= 1 and not wrd in dict_word_to_ignore:
                            if wrd[0] == '#':
                                word = 'médicament'
                            elif wrd == '@card@':
                                word = '1'
                            else:
                                word = wrd
                            if word in dict_voc_already_searched.keys():
                                mean_vect_voc = mean_vect_voc + np.asarray(dict_voc_already_searched[word]).reshape([1,300])
                                nb_word_counted += 1.
                            else:
                                vector_word = get_word(fastext_file, word)
                                if vector_word != []:
                                    dict_voc_already_searched[word] = vector_word
                                    mean_vect_voc = mean_vect_voc + np.asarray(vector_word).reshape([1, 300])
                                    nb_word_counted += 1.
                                else:
                                    dict_word_to_ignore.append(word)

                    mean_vect_voc = mean_vect_voc/float(nb_word_counted)

                    for i in range(300):
                        vect.append(mean_vect_voc[0,i])

                    # writing
                    out.write(ind)
                    print ind
                    for n in vect:
                        out.write(',' + str(n))
                    out.write('\n')

            else:
                first = False
        out.close()




def get_array(namefile):
    """ Return an array from file with index + vector in each line"""
    list_vect = []
    with open(namefile, 'r') as f:
        for line in f:
            vec = []
            new_line = line.replace('\n', '')
            new_line = new_line.split(',')
            for i in range(1, len(new_line)):
                vec.append(float(new_line[i]))
            list_vect.append(vec)
    return np.asarray(list_vect)


# train data
#create_mean_vectors_fromfasttext('data/created/train/input_train_token_normalized.csv', 'data/cc.fr.300.vec', 'data/created/train/vector_input_fasttext2.csv')

# test data
#create_mean_vectors_fromfasttext('data/created/test/input_test_token_normalized.csv', 'data/cc.fr.300.vec', 'data/created/test/vector_input_test_fasttext.csv')
#create_vectors('data/created/input_test_token_normalized.csv', 'data/created/voc_train_second_clean.csv' ,'data/created/vector_test_input.csv')




