


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


#create_list_med_commercialised('data/CIS.txt', 'data/created/med_commercialised.txt')
#create_list_med_non_commercialised('data/CIS.txt', 'data/created/med_non_commercialised.txt')

#print get_list_components('data/COMPO.txt')


