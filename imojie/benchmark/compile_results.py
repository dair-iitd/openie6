import os
import ipdb
import argparse
import regex as re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir')
    return parser

def get_best_models(results_fp):
    accD = dict()
    for line in open(results_fp, 'r'):
        line = line.strip('\n')
        if line == '':
            continue
        if 'pro_output' in line:
            match = re.search('pro_output_(.*).txt', line)
            if match is None:
                ipdb.set_trace()
            epoch_num = int(match.group(1))
            continue
        
        # for optimal point
        # match = re.search('Optimal \(precision, recall, F1\): \( *(.*?), +(.*?), +(.*?),?.*\)', line)
        match = re.search('Optimal \(precision, recall, F1\): \( *(.*?), +(.*?), +(.*?)\)', line)
        if match is None:
            continue
        opt_precision = match.group(1).strip()
        opt_recall = match.group(2).strip()
        opt_f1 = match.group(3).strip()
        
        # for zero confidence point
        match = re.search('Zero Conf \(precision, recall, F1\): \( *(.*?), +(.*?), +(.*?)\)', line)
        if match is None:
            continue
        zero_conf_precision = match.group(1).strip()
        zero_conf_recall = match.group(2).strip()
        zero_conf_f1 = match.group(3).strip()
        
        # for auc
        match = re.search('AUC: (.*) Optimal', line)
        auc = match.group(1)

        try:
            accD[epoch_num] = [float(opt_precision), float(opt_recall), float(opt_f1), float(auc), float(zero_conf_precision), float(zero_conf_recall), float(zero_conf_f1)]
        except:
            ipdb.set_trace()

    best_f1, best_auc, best_sum = -1, -1, -1
    f1_epoch, auc_epoch, sum_epoch = -1, -1, -1
    max_epoch = 0
    for epoch_num in accD:
        f1, auc = accD[epoch_num][2], accD[epoch_num][3]
        if f1 > best_f1:
            best_f1 = f1
            f1_epoch = epoch_num
        if auc > best_auc:
            best_auc = auc
            auc_epoch = epoch_num
        if auc + f1 > best_sum:
            best_sum = auc + f1
            sum_epoch = epoch_num
        if epoch_num > max_epoch:
            max_epoch = epoch_num

    return accD, f1_epoch, auc_epoch, sum_epoch, max_epoch

def main():
    parser = parse_args()
    args = parser.parse_args()
    accD, f1_epoch, auc_epoch, sum_epoch, max_epoch = get_best_models(args.inp_dir + '/results.txt')
    # if f1_epoch != -1:
    #     f1_scores = ['%0.2f'%(s*100) for s in accD[f1_epoch]]
    #     f1_scores_str = '('+', '.join(f1_scores)+', '+str(f1_epoch)+'/'+str(max_epoch)+')'
    #     print('Best F1 Scores = ',f1_scores_str)
    # if auc_epoch != -1:
    #     auc_scores = ['%0.2f'%(s*100) for s in accD[auc_epoch]]
    #     auc_scores_str = '('+', '.join(auc_scores)+', '+str(auc_epoch)+'/'+str(max_epoch)+')'
    #     print('Best AUC Scores = ',auc_scores_str)
    if sum_epoch != -1:
        sum_scores = ['%0.2f'%(s*100) for s in accD[sum_epoch]]
        # sum_scores_str = '('+', '.join(sum_scores)+', '+str(sum_epoch)+'/'+str(max_epoch)+')'
        sum_scores_str = '('+'/'.join(sum_scores[0:3])+', '+str(sum_scores[3])+', '+'/'.join(sum_scores[4:7])+')'
        print('Best Sum Scores = ',sum_scores_str)

    sum_process_fp = args.inp_dir + '/pro_output_'+str(sum_epoch)+'.txt'
    best_process_fp = args.inp_dir + '/best.tsv'
    os.system('cp {} {}'.format(sum_process_fp, best_process_fp))
    sum_model_fp = os.path.join(args.inp_dir,'..')+'/model_state_epoch_'+str(sum_epoch)+'.th'
    best_model_fp = os.path.join(args.inp_dir,'..')+'/best.th'
    if os.path.exists(sum_model_fp):
        os.system('cp {} {}'.format(sum_model_fp, best_model_fp))

    return

if __name__ == '__main__':
    main()

