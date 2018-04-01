from collections import defaultdict

def search_list_word(namefile, list_world):
    phrases = list()
    with open(namefile, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                statement = line[1].split(" ")
                tr = True
                for world in list_world:
                    if world in statement:
                        tr = True and tr
                    else:
                        tr = False and tr
                if tr:
                    phrases.append(line[0])
            nb_line += 1
    return phrases

#mots = ["midi", "matin", "soir"]
#mots = ["##emlapatch##"]
#mots = ["problème"]
mots = ["problème", "##emlapatch##"]
resultats = search_list_word("input_train_token_normalized.csv", mots)
print(resultats)

def count_class(namefile, res):
    dico = defaultdict(int)
    with open(namefile, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if line[0] in res:
                dico[line[1]] += 1
    return dico

solution_file = "challenge_output_data_training_file_predict_the_expected_answer.csv"
dict_sol = count_class(solution_file, resultats)
print(dict_sol)

mots_donnée_base = [
["liste 1", "list 2", "liste II", "liste I"],
["frigo", "frigidaire"],
["generique", "générique", "DCI"],
["secu", "sécurité", "sécu", "securite"],
["tarif", "prix", "coût", "couter", "coute", "coûte", "coût"],
["dose", "doser", "doses"],
["rupture", "stock"],
["dosage", "mesure"],
["bilan"],
["sanguin", "sang", "sanguine", "sanguins", "sanguines"],
["prise"],
["patch", "evra", "elma"],
["Comment", "comment"],
["Ou", "ou", "où", "Où"],
["Combien", "combien"],
["Qu'est", "qu'est"],
["prescription", "prescris", "prescrit", "prescrire", "préscrit"],
["quoi", "Quoi"],
["périmé","perime", "permimés", "perimes"]
]

mots_donnée_norm = [
["substituer", "substitution", "substituable"],
["rembourser"],
["tarif", "prix", "coût", "couter"],
["dose", "doser"],
["alternatif"],
["secondaire"]
]


def add_list_word(namefile_origin, namefile_res, list_dim):
    vects = defaultdict(list)
    with open(namefile_origin, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                statement = line[1].split(" ")
                for list_word in list_dim:
                    tr = False
                    for word in list_word:
                        if word in statement:
                            tr = True
                    if tr:
                        vects[int(line[0])].append(1)
                    else:
                        vects[int(line[0])].append(0)
            nb_line += 1

    with open(namefile_res, 'r') as f_res:
        with open(namefile_res + "_modified", 'w') as f_mod:
            nb = 1
            for line in f_res:
                new_line = line.replace('\n', '')
                for val in vects[nb]:
                    new_line = new_line + "," + str(val)
                new_line = new_line + "\n"
                f_mod.write(new_line)
                nb += 1

add_list_word("input_train.csv", "vector_input_fasttext_only.csv", mots_donnée_base)

def add_size_phrase(namefile_origin, namefile_res):
    vects = defaultdict(int)
    with open(namefile_origin, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                statement = line[1].split(" ")
                vects[int(line[0])] = len(statement)
            nb_line += 1

    with open(namefile_res, 'r') as f_res:
        with open(namefile_res + "_nb", 'w') as f_mod:
            nb = 1
            for line in f_res:
                new_line = line.replace('\n', '')
                new_line = new_line + "," + str(vects[nb])
                new_line = new_line + "\n"
                f_mod.write(new_line)
                nb += 1

add_size_phrase("input_train_norm_medoc_corrected.csv", "vector_input_fasttext_only.csv")
