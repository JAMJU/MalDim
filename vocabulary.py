#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
from medicament import get_list_medoc, is_number, get_list_medoc_first
import enchant
import numpy as np
import os
from lev_distance import levenshtein_dist
import matplotlib.pyplot as plt
import matplotlib as mlab

list_med_com = get_list_medoc_first("data/created/medicaments/med_commercialised.txt")
list_med_non_com = get_list_medoc_first("data/created/medicaments/med_non_commercialised.txt")
dictionnary_french = enchant.Dict('fr_FR')
dict_abreviation = {'qq':"quelque", 'bcp':'beaucoup', 'nb':'nombre', 'qqun':"quelqu'un",'chgt':'changement', 'infos':'informations', 'info':'information' , 'bb':'bébé'}
punctuation = ['"', '.', ',', ';', ':', '/', '!', '?', '(', ')', '#', '*', '<', '>', '[', ']', "'", '€', '-', "’", "=", '+', '&', '`', '%', '»', '«', "'"]
path_tagger = "/home/jamju/Documents/MVA/NLP/TP/Tree_tagger/cmd/tree-tagger-french"
nb = ['1','2','3','4','5','6','7','8','9', '@card@']

def tokenize(tweet):
    """ take a string and tokenize it"""
    # first step : separation by spacing
    new_twt = [tweet]
    next_twt = []
    for wrd in new_twt:
        temp_twt = wrd.split(' ')
        for w in temp_twt:
            if w != '' :
                next_twt.append(w)
    new_twt = next_twt
    # second step : separation by punctuation
    for punct in punctuation:
        if punct in tweet:
            next_twt = []
            for wrd in new_twt:
                temp_twt = wrd.split(punct)
                if len(temp_twt) == 1:
                    next_twt.append(temp_twt[0])
                else:
                    first = True
                    for w in temp_twt:
                        if first:
                            first = False
                        else:
                            next_twt.append(punct)
                        if w != '':
                            next_twt.append(w)
            new_twt = next_twt

    return new_twt

def tokenize_file(namefile, namefile_out):
    """ Re write tokenized version of file nammefile in namefile_out """
    out = codecs.open(namefile_out, 'w')#, 'utf-8')

    with codecs.open(namefile, "r") as f:
        first = True
        for line in f:
            if not first:
                new_line =line.splitlines()[0]
                new_line = line.replace('\n', '')
                new_line = new_line.replace('\r', '')
                new_line = new_line.split(';')
                ind = new_line[0]
                sent = tokenize(new_line[1])
                out.write(ind+ ';')
                for wrd in range(len(sent)):
                    out.write(sent[wrd] + ' ')
                out.write('\n')

            else:
                out.write('ID;question\n')
                first = False
    out.close()

def correction(word):
    """ Correct a mispelled word and return the well spelled version of it"""
    # abreviations case
    if word in 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' '):
        return word
    if word in dict_abreviation.keys():
        return dict_abreviation[word]
    # numbers case
    for k in word:
        if is_number(k):
            return word
    # punctuation case
    if word in punctuation:
        return word
    # medocs case
    if len(word) > 2:
        for med in list_med_com:
            if word in med:
                print "MEDOC ID COM!"
                """medoc = med.split(' ')
                name_medoc = ''
                first = True
                for m in medoc:
                    if m!='':
                        if not first:
                            name_medoc += '-'
                        else:
                            first = False
                        name_medoc += m"""
                return '##' + med + '##'
        for med in list_med_non_com:
            if word in med:
                print "MEDOC ID NON COM!"
                """medoc = med.split(' ')
                name_medoc = ''
                first = True
                for m in medoc:
                    if m != '':
                        if not first:
                            name_medoc += '-'
                        else:
                            first = False
                        name_medoc += m"""
                return '##' + med + '##'


    # use enchant -> if distance reasonable, keep it otherwise we leave it as it is
    suggestions = dictionnary_french.suggest(word)
    suggestions = suggestions[0:min(4, len(suggestions))]
    lev_distances = [levenshtein_dist(word, sug) for sug in suggestions]
    if len(lev_distances) > 1:
        ind = np.argmin(lev_distances)
        if lev_distances[int(ind)] < 3:
            return suggestions[ind]
        else:
            return word
    else:
        return word

def correction_file(namefile, namefile_corrected):
    """ Correct the orthograph of a tokenized file"""
    out = open(namefile_corrected, 'w')
    with open(namefile, 'r') as f:
        for line in f:
            new_line = line.replace('\n', '')
            new_line = new_line.split(';')
            out.write(new_line[0] + ";") # keep the index
            sent = new_line[1].split(' ')

            for wrd in sent:
                if wrd!='' and wrd != '\xc2\xa0' and wrd!=' ':
                    if dictionnary_french.check(wrd.lower()):
                        out.write(wrd.lower() + " ")
                    else:
                        out.write(correction(wrd.lower()) + " ")
            out.write("\n")
    out.close()


def normalize_word(word):
    """ normalize the word using tree tagger, ie if adj transform it in neutral, if verb unconjugate it..."""
    if word != '' and word != ' ':
        if "'" in word:
            word = '"' + word + '"'
        bashCommand = "echo " + word + " |" + path_tagger + " > temp_file.txt"
        os.system(bashCommand)
        file_temp = open("temp_file.txt")
        list_lines = []
        for line in file_temp:
            list_lines.append(line)
        file_temp.close()

        line1 = list_lines[0].replace("\n", '')
        wrd = line1.split('\t')[2]
        with open("temp_file.txt", "w"):
            pass

        # first test if unknown:
        test = list_lines[0].split("<")
        if len(test) > 1:
            test = test[1].split(">")[0]
            if test== "unknown":
                return word

        else:
            return ' '.join(wrd.split('|'))
    else:
        return word

def normalize_file(namefile, file_normalized):
    "Normalize the sentences in order to extract the right vocabulary"
    out = open(file_normalized, 'w')
    with open(namefile, 'r') as f:
        for line in f:
            new_line = line.replace('\n', '')
            new_line = new_line.split(';')
            out.write(new_line[0] + ";")  # keep the index
            sent = new_line[1].split(' ')

            for wrd in sent:
                if wrd != '' and wrd != '\xc2\xa0' and wrd != ' ' :
                    if not wrd in punctuation and wrd[0] != '#':
                        out.write(normalize_word(wrd))
                        out.write(' ')
                    else:
                        out.write(wrd)
                        out.write(' ')
            out.write("\n")
    out.close()

def get_vocabulary_list_and_scores(namefile_tokenize, namefile_voc):
    """ From a tokenized file, get list of apparition of word in the all document"""
    out = codecs.open(namefile_voc, 'w')
    dico_voc = dict()
    with codecs.open(namefile_tokenize, 'r') as f:
        for line in f:
            new_line = line.replace('\n', '')
            new_line = new_line.split(";")
            new_line = new_line[1].split(' ')
            for wrd in new_line:
                if wrd in dico_voc.keys():
                    dico_voc[wrd] += 1
                else:
                    dico_voc[wrd] = 1

    for wrd in dico_voc.keys():
        out.write(wrd + ";" + str(dico_voc[wrd]))
        out.write('\n')
    out.close()
    return dico_voc

def clear_vocabulary(namefile_in, namefile_out):
    """ Get a file withe vocabulary and nb of apparitions and create a new file with vocabulary and apparitions but
    with medoc, punctuation an lettres alone removed"""
    out = open(namefile_out, 'w')
    with open(namefile_in, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
            else:
                new_line = line.split(';')
                qt = int(new_line[1].replace('\n', ''))
                new_line = new_line[0]

                numbersIn = False
                for n in nb:
                    if n in new_line:
                        numbersIn = True
                punct = False
                for n in punctuation:
                    if n in new_line:
                        punct = True

                if line[0] != '#' and not numbersIn and not punct and qt>=4 and len(new_line) > 1:
                    out.write(line)
    out.close()

def analyse(namefile_in):
    """ Print an histogram of the word from a voc file"""
    quant = []
    with open(namefile_in, 'r') as f:
        for line in f:

            new_line = line.split(';')
            qt = int(new_line[1].replace('\n', ''))
            quant.append(qt)

    # the histogram of the data
    n, bins, patches = plt.hist(quant, 200,  facecolor='green', alpha=0.75)
    plt.show()

def get_vocabulary_list(namefile):
    """ Take a voc file and return a list of word"""
    list_voc = []
    with open(namefile, 'r') as f:
        for line in f:
            li = line.replace('\n', '')
            li = li.split(';')
            if not li[0] in punctuation and not ':' in li[0]:
                list_voc.append(li[0])
    return list_voc

def get_vocabulary_list_appear(namefile):
    """ Take a voc file and return a list of word"""
    list_voc = dict()
    with open(namefile, 'r') as f:
        for line in f:
            li = line.replace('\n', '')
            li = li.split(';')
            if not li[0] in punctuation and not ':' in li[0]:
                list_voc[li[0]] = li[1]
    return list_voc

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
    for line in file:
        new_line = line.replace('\n', '')
        sent = new_line.split(' ')[1]
        list_sent.append(sent)
    return list_sent

def get_vocabulary_present_in_class(label, namefile_voc, namefile_input, namefile_label, namefile_out):
    """ From file of voc, create a file with the voc present in senteneces if their label is the one selected"""
    voc_list = get_vocabulary_list(namefile_voc)
    list_sent = get_sentences_list(namefile_input)
    list_label = get_label_list(namefile_label)
    voc_present = dict()
    nb = 0
    for i in range(len(list_sent)):
        if list_label[i] == label:
            nb += 1
            sent_listed = list_sent[i].split(' ')
            for voc in voc_list:
                #print voc
                if voc in sent_listed:
                    if voc in voc_present.keys():
                        voc_present[voc] += 1
                    else:
                        voc_present[voc] = 1
    out = open(namefile_out, 'w')
    out.write("nb of sentences :" + str(nb) + '\n')
    for wrd in voc_present.keys():
        out.write(wrd + ";" + str(voc_present[wrd]) + '\n')

def compare_voc_present(namefile_voc, namefile_typ, nb_class_appear, namefile_out):
    """
    :param namefile_voc:
    :param namefile_typ:
    :param nb_class_appear:
    :param namefile_out:
    :return: write a file with all the vocabulary present in nb_class appear class
    """
    voc_list = get_vocabulary_list(namefile_voc)
    dict_nb_class = {voc:[] for voc in voc_list}
    dict_nb_appear = {voc:0 for voc in voc_list}
    for i in range(51):
        voc = get_vocabulary_list_appear(namefile_typ + str(i) + '.csv')
        for wrd in voc_list:
            if wrd in voc.keys():
                dict_nb_class[wrd].append(int(voc[wrd]))
                dict_nb_appear[wrd] += 1
            else:
                dict_nb_class[wrd].append(0)
    out = open(namefile_out, 'w')
    for wrd in dict_nb_class.keys():
        if dict_nb_appear[wrd] == nb_class_appear:
            out.write(wrd + ";")
            for k in range(len(dict_nb_class[wrd])):
                out.write(" " + str(dict_nb_class[wrd][k]))
            out.write("\n")
    out.close()

def get_voc_compare(namefile_type, namefile_out, beg, ending):
    """ Write in a file all voc present from beg types of classe to ending types of class"""
    out= open(namefile_out, 'w')
    for i in range(beg, ending + 1):
        voc = get_vocabulary_list(namefile_type + str(i) + ".csv")
        for wrd in voc:
            out.write(wrd + "\n")
    out.close()







# train data

#tokenize_file('data/input_train.csv', 'data/created/input_train_token.csv')
#correction_file('data/created/train/input_train_token.csv', 'data/created/input_train_token_corrected_v2.csv')
#correction_file('data/created/test/input_test_token.csv', 'data/created/input_test_token_corrected_v2.csv')
#normalize_file('data/created/train/input_train_token_corrected_v2.csv', 'data/created/train/input_train_token_nor_v2.csv')
#normalize_file('data/created/test/input_test_token_corrected_v2.csv', 'data/created/test/input_test_token_nor_v2.csv')
#get_vocabulary_list_and_scores("data/created/input_train_token_normalized.csv", "data/created/vocabulary_train.csv")
#clear_vocabulary( "data/created/vocabulary_train.csv",  "data/created/vocabulary_train_firstclean.csv")
#analyse("data/created/vocabulary_train_firstclean.csv")
#for i in range(52):
#    get_vocabulary_present_in_class(i,"data/created/vocabulary_train_firstclean.csv", 'data/created/input_train_token_normalized.csv', 'data/label.csv', 'data/created/voc_present_classes/voc_labl'+ str(i) + '.csv' )
#for i in range(1,52):
#    compare_voc_present("data/created/vocabulary_train_firstclean.csv", 'data/created/voc_present_classes/voc_labl', i, 'data/created/voc_present_nb_classes/' + str(i) + '.csv')
#get_voc_compare('data/created/voc_present_nb_classes/' , 'data/created/voc_train_second_clean.csv', 1,  37)

# test data
#tokenize_file('data/input_test.csv', 'data/created/input_test_token.csv')
#correction_file('data/created/input_test_token.csv', 'data/created/input_test_token_corrected.csv')
#normalize_file('data/created/input_test_token_corrected.csv', 'data/created/input_test_token_normalized.csv')