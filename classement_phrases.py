from collections import defaultdict

def get_phrase_id(namefile):
    phrase_id = dict()
    with open(namefile, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                phrase_id[int(line[0])] = int(line[1])
            nb_line += 1
    return phrase_id

solution_file = "challenge_output_data_training_file_predict_the_expected_answer.csv"
phrase_id = get_phrase_id(solution_file)

def get_dico_classe(namefile, phrase_id):
    dico_classe = defaultdict(list)
    with open(namefile, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                dico_classe[phrase_id[int(line[0])]].append(line[1])
            nb_line += 1
    return dico_classe

dict_classes = get_dico_classe("input_train.csv", phrase_id)

def create_class_files(namefold, dict_classes):
    for key in dict_classes:
        with open(namefold + "classe_" +  str(key) + ".csv", 'w') as f:
            for phr in dict_classes[key]:
                f.write(phr)
                f.write("\n")

#create_class_files("classement/", dict_classes)
