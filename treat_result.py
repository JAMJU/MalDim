def treat_results(result_file, out_file, begin_index):
    out = open(out_file, 'w')
    ind = begin_index
    out.write('ID;intention\n')
    with open(result_file, 'r') as f:
        for line in f:
            new_line = line.replace('\n', '')
            out.write(str(ind) + ';' + new_line)
            out.write('\n')
            ind +=1


treat_results('data/created/results/log_reg_fasttext_and_other_v2_raw_C-500.csv', 'data/created/results/log_reg_fasttext_and_other_v2_C-500.csv', 8028)


