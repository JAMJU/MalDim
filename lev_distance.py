import numpy as np

def levenshtein_dist(word1, word2):
    ref_mat = np.zeros([len(word1) + 1, len(word2) + 1])
    for i in range(len(word1) + 1):
        ref_mat[i, 0] = i
    for j in range(len(word2) + 1):
        ref_mat[0,j] = j

    for i in range(1,len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                ref_mat[i,j] = min([ref_mat[i-1,j] + 1, ref_mat[i, j-1] + 1, ref_mat[i-1, j-1]])
            else:
                ref_mat[i, j] = min([ref_mat[i - 1, j] + 1, ref_mat[i, j - 1] + 1, ref_mat[i - 1, j - 1] + 1])
    return ref_mat[len(word1), len(word2)]