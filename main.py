#from kernel_creation import get_array
#from vocabulary import get_label_list
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.colors as mpc
colors = mpc.cnames.keys()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from study_data import compare_pred_truth
from collections import Counter

def get_array(namefile):
    """ Return an array from file with index + vector in each line"""
    list_vect = []
    first = True
    len_sup = 0
    with open(namefile, 'r') as f:
        for line in f:
            vec = []
            new_line = line.replace('\n', '')
            new_line = new_line.split(',')
            for i in range(1, len(new_line)):
                vec.append(float(new_line[i]))
            if first:
                list_vect.append(vec)
                len_sup = len(vec)
                first = False
            else:
                if len_sup == len(vec):
                    list_vect.append(vec)
    return np.asarray(list_vect)

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

def write_label(label_list, out_file, begin_index):
    out = open(out_file, 'w')
    ind = begin_index
    out.write('ID;intention\n')

    for i in range(len(label_list)):
        out.write(str(ind) + ';' +str(label_list[i]))
        out.write('\n')
        ind += 1


# Choose the model
#
Log_reg = True
C_log = 500.



mat = get_array( 'data/created/train/vector_input_fasttext.csv')
mat_test = get_array('data/created/test/vector_input_test_fasttext.csv')
lab = get_label_list('data/label.csv')
color_lab = [colors[l] for l in lab]

nb_cross_validation = 10
size_test = int(float(mat.shape[0])/float(nb_cross_validation))
print("size test", size_test)
beg = 0
total_mis_pred = {k:0 for k in range(51)}
total_good_pred = {k:0 for k in range(51)}
class_instead = {k:[] for k in range(51)}
mean_train = 0.
mean_test = 0.
for i in range(nb_cross_validation):
    print("cross validation ", i, "on", nb_cross_validation)
    mat_test_train = mat[beg:beg + size_test, :]
    lab_test_train = lab[beg : beg + size_test]

    mat_train = np.concatenate((mat[0:beg], mat[beg + size_test:, :]), axis = 0)
    lab_train = lab[0:beg] + lab[beg + size_test:]
    beg = beg + size_test

    ## Classification method
    if Log_reg:
        from sklearn.linear_model import LogisticRegression

        # instantiate a logistic regression model, and fit with X and y
        model = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=C_log, verbose=1)
        model = model.fit(mat_train, lab_train)

        # check the accuracy on the training set
        score_train =  model.score(mat_train, lab_train)
        score_test = model.score(mat_test_train, lab_test_train)
        print ("On training",score_train)
        print ("On test", score_test)
        mean_train += score_train/float(nb_cross_validation)
        mean_test += score_test/float(nb_cross_validation)

        y = model.predict(mat_test_train)
        dict_rep, dict_true, dict_false, dict_id = compare_pred_truth(lab_test_train, y)
        for k in dict_rep.keys():
            #print ("classe", k, "nb:", dict_rep[k], "good_pred:", dict_true[k], "fake_pred:", dict_false[k])
            #print (" ")
            total_mis_pred[k] += dict_false[k]
            total_good_pred[k] += dict_true[k]
            class_instead[k] += dict_id[k]

out = open("data/created/result_log_reg_500_fasttext.txt", 'w')
for k in range(51):
    print("class", k, "good pred", total_good_pred[k], "fake pred", total_mis_pred[k], "ok percent", float(total_good_pred[k])/float(total_mis_pred[k] + total_good_pred[k]))
    out.write("class " + str(k) + " good pred " + str(total_good_pred[k]) + " fake pred " + str(total_mis_pred[k]) + " ok percent " + str( float(total_good_pred[k])/float(total_mis_pred[k] + total_good_pred[k])))
    out.write('\n')
    out.write('\n')
    print(" ")
    print("IDENTIFIED INSTEAD OF CLASS", k, '#######################################"')
    out.write("IDENTIFIED INSTEAD OF CLASS "+ str( k) +  '#######################################' + '\n')
    print(Counter(class_instead[k]))
    out.write(str(Counter(class_instead[k])) + '\n')
    print('###########################################################')
    out.write('###########################################################' + '\n')
print("TOTAL", "train", mean_train, "test", mean_test)
out.write("TOTAL " + "train " + str(mean_train) +  " test " + str(mean_test))






# Ridge classifier # not bad -> to test with all the data
"""from sklearn.linear_model import RidgeClassifier
model_ridge = RidgeClassifier(normalize=True, alpha = 0.00001)
model_ridge.fit(mat_train, lab_train)
print "on training", model_ridge.score(mat_train, lab_train)
print "on test training", model_ridge.score(mat_test_train, lab_test_train)"""
"""g = open('data/created/test_train/original.csv', 'w')
for i in range(len(lab_test_train)):
    g.write(str(i)+ ';'+ str(lab_test_train[i])+ '\n' )
f = open('data/created/test_train/results.csv', 'w')
y = model_ridge.predict(mat_test_train)
for i in range(len(y)):
    f.write(str(i) + ';' + str(y[i]) + "\n")
f.close()"""

"""f = open('data/created/results/ridge_reg_fasttext_raw.csv', 'w')
y = model_ridge.predict(mat_test)
for i in range(len(y)):
    f.write(str(y[i]) + "\n")
f.close()"""

"""# Ridge classifier with cv
from sklearn.linear_model import RidgeClassifierCV
model_ridge = RidgeClassifierCV(normalize=True, alphas = np.asarray([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]), cv = 5)

model_ridge.fit(mat_train, lab_train)

print "on training", model_ridge.score(mat_train, lab_train)
print "on test training", model_ridge.score(mat_test_train, lab_test_train)"""


# Logistic Regression 
"""from sklearn.linear_model import LogisticRegression


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression(solver = 'newton-cg', multi_class='multinomial', C=500., verbose=1)
model = model.fit(mat_train, lab_train)

# check the accuracy on the training set
print "On training", model.score(mat_train, lab_train)
print "On test", model.score(mat_test_train, lab_test_train)

f = open('data/created/results/log_reg_fasttext_raw_C-500.csv', 'w')
y = model.predict(mat_test)
for i in range(len(y)):
    f.write(str(y[i]) + "\n")
f.close()"""


# Boosting
"""from sklearn.ensemble import GradientBoostingClassifier
print "beginning boosting"
gb = GradientBoostingClassifier(verbose=1, max_depth=3)
gb.fit(mat_train, lab_train)
print "on training", gb.score(mat_train, lab_train)
print "on test train", gb.score(mat_test_train, lab_test_train)
y = gb.predict(mat_test)

f = open('data/created/results_boost.csv', 'w')
for res in y:
    f.write(str(res) + "\n")
f.close()"""

"""# Decision Tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(mat_train, lab_train)
print "score train", clf.score(mat_train, lab_train)
print "score test", clf.score(mat_test_train, lab_test_train)
#print cross_val_score(clf, mat, lab[0:mat.shape[0]], cv=3) # to do cross validation"""

"""# Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=15, random_state=0)
clf.fit(mat_train, lab_train)
print "score training", clf.score(mat_train, lab_train)
print "score test train", clf.score(mat_test_train, lab_test_train)"""

# network
"""from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(200, activation='relu', input_dim=mat_train.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(51, activation='softmax'))
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(lab_train, num_classes=51)

# Train the model, iterating on the data in batches of 32 samples
model.fit(mat_train, one_hot_labels, epochs=200, batch_size=100)


one_hot_labels_test =  to_categorical(lab_test_train, num_classes=51)
print one_hot_labels_test
score_train = model.evaluate(mat_train, one_hot_labels, batch_size=100)
print score_train
score = model.evaluate(mat_test_train, one_hot_labels_test, batch_size=100)
print score
result = model.predict( mat_test, batch_size=100, verbose=0, steps=None)
result = np.argmax(result, axis = 1)
g = open('data/created/test_train/original_net.csv', 'w')
for i in range(len(lab_test_train)):
    g.write(str(i)+ ';'+ str(lab_test_train[i])+ '\n' )
f = open('data/created/results/results_net_200_1.csv', 'w')
for i in range(len(result)):
    #f.write(str(i) + ';' + str(result[i]) + "\n")
    f.write(str(result[i]) + "\n")
f.close()"""





