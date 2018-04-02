#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def levenshtein_dist(word1, word2):
    word1_fix = word1.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('â', 'a').replace('ô', 'o').replace('ë', 'e').replace('ï', 'i').replace('û', 'u')
    word2_fix = word2.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('â', 'a').replace('ô', 'o').replace('ë', 'e').replace('ï', 'i').replace('û', 'u')
    ref_mat = np.zeros([len(word1_fix) + 1, len(word2_fix) + 1])
    for i in range(len(word1_fix) + 1):
        ref_mat[i, 0] = i
    for j in range(len(word2_fix) + 1):
        ref_mat[0,j] = j

    for i in range(1,len(word1_fix) + 1):
        for j in range(1, len(word2_fix) + 1):
            if word1_fix[i - 1] == word2_fix[j - 1]:
                ref_mat[i,j] = min([ref_mat[i-1,j] + 1, ref_mat[i, j-1] + 1, ref_mat[i-1, j-1]])
            else:
                ref_mat[i, j] = min([ref_mat[i - 1, j] + 1, ref_mat[i, j - 1] + 1, ref_mat[i - 1, j - 1] + 1])
    return ref_mat[len(word1_fix), len(word2_fix)]


