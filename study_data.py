

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

def get_sentences_list(namefile):
    """ Return the list of sentences in namfile"""
    list_sent = []
    first = True
    with open(namefile, 'r') as f:
        for line in f:
            if not first:
                new_line = line.split(';')
                qt = new_line[1].replace('\n', '')
                list_sent.append(qt)
            else:
                first = False
    return list_sent

def get_repartition_label(list_label):
    repartition = {}
    for i in range(51):
        repartition[i] = list_label.count(i)
    return repartition

def compare_pred_truth(list_label_true, list_label_pred):
    dict_repart = get_repartition_label(list_label_true)
    dict_true = {i:0 for i in range(51)}
    dict_false = {i:0 for i in range(51)}
    for i in range(len(list_label_true)):
        if list_label_true[i] == list_label_pred[i]:
            dict_true[list_label_true[i]]  +=1
        else:
            dict_false[list_label_true[i]] += 1
    return dict_repart, dict_true, dict_false

label_all = get_label_list("data/label.csv")
dict_label_all = get_repartition_label(label_all)
list_label_true = get_label_list("data/created/test_train/original_net.csv")
list_label_pred = get_label_list("data/created/test_train/results_net.csv")
dict_rep, dict_true, dict_false = compare_pred_truth(list_label_true, list_label_pred)
for k in dict_rep.keys():
    print "classe", k, "nb:", dict_rep[k], "total in database", dict_label_all[k], "good_pred:", dict_true[k], "fake_pred:", dict_false[k]
    print " "


