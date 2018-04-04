#!/usr/bin/env python
# -*- coding: utf-8 -*-
from lev_distance import levenshtein_dist
import enchant
dictionnary_french = enchant.Dict('fr_FR')
dictionnary_french.add('libido')
dictionnary_french.add( 'eczema')
dictionnary_french.add('crohn')
dictionnary_french.add('gynéco')
dictionnary_french.add('week')
dictionnary_french.add('deja')
dictionnary_french.add('bientot')

import numpy as np
punctuation = ['"', '.', ',', ';', ':', '/', '!', '?', '(', ')', '#', '*', '<', '>', '[', ']', "'", '€', '-', "’", "=", '+', '&', '`', '%', '»', '«', "'", "’"]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def create_list_med_commercialised(namefile, namefile_out):
    """from a list of medoc. create list of medoc commercialised"""
    out = open(namefile_out, 'w')
    with open(namefile, 'r') as f:
        for line in f:
            line_ = line.replace('\n', '')
            line_ = line_.split(' ')
            # We check the med is commercialised
            keep = False
            if not "Non" in line_:
                keep = True
            else:
                if not line_[line_.index("Non") + 1][0:3] == "com" :
                    keep=True
            if keep: # if it is we get its name
                new_line = line.split(',')
                name_ap = new_line[0]
                name_list = name_ap.split(' ')
                ind = name_list[0] # the identification number of the med
                out.write(ind)
                out.write(';')
                nb_found = False
                first = True
                for n in name_list[1:len(name_list)]:
                    if is_number(n) and not nb_found and not first:
                        nb_found = True
                        out.write(';')
                    if first:
                        first = False
                    out.write(n.lower() + " ")
                out.write('\n')

def create_list_med_non_commercialised(namefile, namefile_out):
    """idem than last function but with non commercialised medoc."""
    out = open(namefile_out, 'w')
    with open(namefile, 'r') as f:
        for line in f:
            line_ = line.replace('\n', '')
            line_ = line_.split(' ')
            # We check the med is commercialised
            keep = False
            if "Non" in line_:
                if line_[line_.index("Non") + 1][0:3] == "com" :
                    keep=True

            if keep: # if it is not we get its name
                new_line = line.split(',')
                name_ap = new_line[0]
                name_list = name_ap.split(' ')
                ind = name_list[0] # the identification number of the med
                out.write(ind)
                out.write(';')
                nb_found = False
                first = True
                for n in name_list[1:len(name_list)]:
                    if is_number(n) and not nb_found and not first:
                        nb_found = True
                        out.write(';')
                    if first:
                        first = False
                    out.write(n.lower() + " ")
                out.write('\n')

def get_list_medoc(namefile):
    """ Get the list of the medic. in a file like the ones created by the functions above"""
    list_medoc = []
    with open(namefile, 'r') as f:
        for line in f:
            new_line = line.split(';')
            if len(new_line)> 1:
                list_medoc.append(new_line[1].replace('\n', ''))
    return list_medoc

def get_list_medoc_first(namefile):
    """ Get the list of the medic. in a file like the ones created by the functions above but just first name"""
    list_medoc = []
    with open(namefile, 'r') as f:
        for line in f:
            new_line = line.split(';')
            if len(new_line) > 1:
                list_medoc.append(new_line[1].replace('\n', '').split(' ')[0])
    return list_medoc

def get_list_components(namefile):
    """ Get list of components in medoc list : COMPO.txt"""
    list_comp =[]
    with open(namefile, 'r') as f:
        for line in f:
            new_line = line.replace('\n', '')
            new_line = new_line.split('	')
            compo = new_line[2]
            if not compo in list_comp:
                list_comp.append(list_comp)

    return len(list_comp)

def get_medicamnt_close(list_medoc, name_to_look_for):
    """ Return the name of medoc closer un list_medoc"""
    print(name_to_look_for)
    if dictionnary_french.check(name_to_look_for.lower()): # just recheck
        return name_to_look_for
    else:
        list_first_letter_same = [list_medoc[i] for i in range(len(list_medoc)) if list_medoc[i][0].lower() == name_to_look_for[0].lower() ]
        if list_first_letter_same == []:
            return name_to_look_for
        lev_dist = []
        for i in range(len(list_first_letter_same)):
            lev_dist.append(levenshtein_dist(name_to_look_for, list_first_letter_same[i]))
        dist = min(lev_dist)
        if dist <= 2:
            ind = int(np.argmin(lev_dist))
            return list_first_letter_same[ind]
        else:
            return name_to_look_for

def write_new_file_with_medoc_rewritten(namefile_input, namefile_out, list_medoc):
    """ Write namefile_out with the name of the medoc better identified"""
    input = open(namefile_input, 'r')
    output = open(namefile_out, 'w')
    nb = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    first = True
    count = 0
    for line in input:
        print(count)
        count +=1
        if not first:
            new_line = line.replace('\n', '')
            new_line = new_line.split(';')
            ind = new_line[0]
            sent = new_line[1].split(' ')
            new_sent = []
            output.write(ind + ';')
            for j in range(len(sent)):
                if len(sent[j]) >= 1:
                    if sent[j] in punctuation:
                        new_sent.append(sent[j])
                    elif sent[j][0] == "#":
                        name_med = sent[j].split('#')[2]
                        name_med = name_med.split(' ')[0]
                        new_sent.append('##' + name_med+ '##')
                    elif sent[j][0] == '@':
                        new_sent.append(sent[j])
                    else:
                        nb_in = False
                        for k in nb:
                            if str(k) in sent[j] and not nb_in:
                                new_sent.append(sent[j])
                                nb_in = True
                        if not nb_in:
                            if dictionnary_french.check(sent[j]) or len(sent[j]) <= 2:
                                new_sent.append(sent[j])
                            else:
                                med= get_medicamnt_close(list_medoc, sent[j])
                                print(sent[j], med)
                                if med != sent[j]:
                                    new_sent.append('##' + med + '##')
                                else:
                                    new_sent.append(med)


            output.write(" ".join(new_sent) + '\n')
        elif first:
            first = False
            output.write("ID;question" + '\n')

def write_list_medoc_in_file(filename_input, filename_output):
    out = open(filename_output, 'w')
    list_medoc = []
    with open(filename_input, 'r') as f:
        for line in f:
            new_line = line.split(';')
            new_line = new_line[1].replace('\n', '')
            new_line = new_line.split(' ')
            for wrd in new_line:
                if len(wrd) > 1:
                    if wrd[0] == '#':
                        medoc = wrd.split('#')[2]
                        if not medoc in list_medoc:
                            list_medoc.append(medoc)
    for med in list_medoc:
        out.write(med + '\n')






#create_list_med_commercialised('data/CIS.txt', 'data/created/med_commercialised.txt')
#create_list_med_non_commercialised('data/CIS.txt', 'data/created/med_non_commercialised.txt')

#print get_list_components('data/COMPO.txt')

"""list_med_com = get_list_medoc_first('data/created/medicaments/med_commercialised.txt')
list_med_uncom = get_list_medoc_first('data/created/medicaments/med_non_commercialised.txt')
list_total_med = list_med_com + list_med_uncom
write_new_file_with_medoc_rewritten('data/created/train/input_train_token_nor_v2.csv', 'data/created/train/input_train_norm_medoc_corrected_v2.csv', list_total_med )
write_new_file_with_medoc_rewritten('data/created/test/input_test_token_nor_v2.csv', 'data/created/test/input_test_norm_medoc_corrected_v2.csv', list_total_med )"""

# write_list_medoc_in_file("../input_test_norm_medoc_corrected_v2.csv",  "../test_v2_list_medoc.csv")
