# -*- coding: utf-8 -*-
import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def get_word(fname, word):
    """return rep vector of word according to fasttext """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    #n, d = map(int, fin.readline().split())

    for line in fin:
        tokens = line.rstrip().split(' ')

        if tokens[0]== word.decode('utf8'):
            print word
            print 'OK'
            return map(float, tokens[1:])
    print word
    print 'NOT OK'
    return []