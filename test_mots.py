#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np

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
# mots = ["problème", "##emlapatch##"]
# resultats = search_list_word("input_train_token_normalized.csv", mots)
# print(resultats)

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

# solution_file = "challenge_output_data_training_file_predict_the_expected_answer.csv"
# dict_sol = count_class(solution_file, resultats)
# print(dict_sol)

# mots_donnée_base = [
# ["liste 1", "list 2", "liste II", "liste I"],
# ["frigo", "frigidaire"],
# ["generique", "générique", "DCI"],
# ["secu", "sécurité", "sécu", "securite"],
# ["tarif", "prix", "coût", "couter", "coute", "coûte", "coût"],
# ["dose", "doser", "doses"],
# ["rupture", "stock"],
# ["dosage", "mesure"],
# ["bilan"],
# ["sanguin", "sang", "sanguine", "sanguins", "sanguines"],
# ["prise"],
# ["patch", "evra", "elma"],
# ["Comment", "comment"],
# ["Ou", "ou", "où", "Où"],
# ["Combien", "combien"],
# ["Qu'est", "qu'est"],
# ["prescription", "prescris", "prescrit", "prescrire", "préscrit"],
# ["quoi", "Quoi"],
# ["périmé","perime", "permimés", "perimes"]
# ]

mots_donnée_norm = [
["substituer", "substitution", "substituable"],
["générique", "generique"],
["secable", "insécable", "coupable"],
["couper"],
["gélule", "##arkogelules##"],
["danger", "dangereux"],
["risque", "risquer"],
["nocif"],
["rembourser", "remboursement", "remboursable"],
["sécu"],
["charge"],
["tarif", "prix"],
["coût", "coûter"],
["cher"],
["##adcirca##"],
["grocesse", "enceinte"],
["nourisson", "bébé"],
["dosage", "dose", "doser"],
["posologie"],
["par"],
["jour", "semaine", "mois", "année"],
["métabolisation", "métabolisme"],
["élimination", "éliminer"],
["temps"],
["naturel"],
["origine"],
["moment"],
["soir"],
["matin"],
["midi"],
["heure"],
["prendre"],
["mélanger", "mélange"],
["diluer", "dilution"],
["comment"],
["quand"],
["combien"],
["alcool"],
["soleil"],
["cannabis"],
["cigarette"],
["compatible"],
["quoi"],
["ou"],
["où"],
["traitement"],
["traiter"],
["pourquoi"],
["quel"],
["forme"],
["suppositoire"],
["sirop"],
["comprimer"],
["exister"],
["acheter"],
["alternatif"],
["secondaire"],
["durée"],
["pendant"],
["depuis"],
["effet"],
["trouver"],
["disponible"],
["pharmacie"],
["marché"],
["sevrage", "sevrer"],
["arrêt", "arrêter"],
["marque"],
["péremption", "périmer"],
["vaccin"],
["frigo", "frigidaire", "réfrigérateur"],
["température"],
["réchauffer", "chaud"],
["ouvrir", "ouverture"],
["conserver", "conservation"],
["##emlapatch##", "##evra##"],
["plaquette"],
["pilule"],
["oublier", "oubli", "oublie"],
["passage", "passer"],
["continuer"],
["changer", "changement"],
["prescrire", "prescription"],
["ordonnance"],
["sans"],
["lister", "liste"],
["flacon"],
["contenir"],
["composition", "composer"],
["fabriquant"],
["rupture"],
["stock"],
["manque", "manquer"],
["bilan"],
["sang", "sanguin"],
["prise"],
["remplacer", "remplacement"]
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
                        vects[nb_line].append(1)
                    else:
                        vects[nb_line].append(0)
            nb_line += 1

    with open(namefile_res, 'r') as f_res:
        split_res_file = namefile_res.split(".")
        rewrite = split_res_file[0] + "_modified." + split_res_file[1]
        with open(rewrite, 'w') as f_mod:
            nb = 1
            for line in f_res:
                new_line = line.replace('\n', '')
                for val in vects[nb]:
                    new_line = new_line + "," + str(val)
                new_line = new_line + "\n"
                f_mod.write(new_line)
                nb += 1

# add_list_word("input_train_norm_medoc_corrected_v2.csv", "vector_input_fasttext_and_other_v2.csv", mots_donnée_norm)

def add_size_phrase(namefile_origin, namefile_res):
    vects = defaultdict(int)
    with open(namefile_origin, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                statement = line[1].split(" ")
                vects[nb_line] = len(statement)
            nb_line += 1

    with open(namefile_res, 'r') as f_res:
        split_res_file = namefile_res.split(".")
        rewrite = split_res_file[0] + "_nb." + split_res_file[1]
        with open(rewrite, 'w') as f_mod:
            nb = 1
            for line in f_res:
                new_line = line.replace('\n', '')
                if vects[nb] < 5:
                    new_line = new_line + "," + "1,0,0,0,0,0,0"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                elif vects[nb] < 10:
                    new_line = new_line + "," + "0,1,0,0,0,0,0"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                elif vects[nb] < 15:
                    new_line = new_line + "," + "0,0,1,0,0,0,0"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                elif vects[nb] < 20:
                    new_line = new_line + "," + "0,0,0,1,0,0,0"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                elif vects[nb] < 30:
                    new_line = new_line + "," + "0,0,0,0,1,0,0"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                elif vects[nb] < 50:
                    new_line = new_line + "," + "0,0,0,0,0,1,0"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                else:
                    new_line = new_line + "," + "0,0,0,0,0,0,1"
                    new_line = new_line + "\n"
                    f_mod.write(new_line)
                nb += 1

#add_size_phrase("input_test_norm_medoc_corrected_v2.csv", "vector_input_test_fasttext_and_other_v2_modified.csv")

def get_medoc_used(namefile_med, namefile_data):
    medocs = defaultdict(int)
    name_medocs = list()
    with open(namefile_med, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            name = "##" + str(line) + "##"
            medocs[name] = 0
            name_medocs.append(name)

    with open(namefile_data, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                statement = line[1].split(" ")
                for med_name in name_medocs:
                    if med_name in line[1]:
                        medocs[med_name] += 1
            nb_line += 1

    s_nb = 0
    final_list = list()
    for med_name in name_medocs:
        if medocs[med_name] > 10:
            s_nb += 1
            final_list.append(med_name)
    print(s_nb)
    print(final_list)
    return final_list

selected_list = get_medoc_used("train_v2_list_medoc.csv", "input_train_norm_medoc_corrected_v2.csv")

def add_list_medocs(namefile_origin, namefile_res, list_medocs):
    vects = defaultdict(list)
    with open(namefile_origin, 'r') as f:
        nb_line = 0
        for line in f:
            line = line.replace('\n', '')
            line = line.split(';')
            if nb_line != 0:
                statement = line[1].split(" ")
                for med in list_medocs:
                    tr = False
                    if med in statement:
                        tr = True
                    if tr:
                        vects[nb_line].append(1)
                    else:
                        vects[nb_line].append(0)
            nb_line += 1

    with open(namefile_res, 'r') as f_res:
        split_res_file = namefile_res.split(".")
        rewrite = split_res_file[0] + "_meds." + split_res_file[1]
        with open(rewrite, 'w') as f_mod:
            nb = 1
            for line in f_res:
                new_line = line.replace('\n', '')
                for val in vects[nb]:
                    new_line = new_line + "," + str(val)
                new_line = new_line + "\n"
                f_mod.write(new_line)
                nb += 1

# add_list_medocs("input_test_norm_medoc_corrected_v2.csv", "vector_input_test_fasttext_and_other_v2_modified_nb.csv", selected_list)
