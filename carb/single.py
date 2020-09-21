import os
import sys
import ipdb
import regex
import random
from tqdm import tqdm
from distutils.util import strtobool
import argparse
import math
import numpy as np

import carb

random.seed(1234)
global args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fp', type=str)
    parser.add_argument('--out_fp', type=str)
    parser.add_argument('--gold_fp', type=str)

    return parser

def run_carb(inp_fp, out_fp, gold_fp):
    inp_f = open(inp_fp,'r')
    out_f = open(out_fp,'w')
    
    input_lines = inp_f.readlines()
    b = carb.Benchmark(gold_fp)
    out_filename = "/dev/null"
    matchingFunc = carb.Matcher.binary_linient_tuple_match

    predicted = carb.AllennlpReader(threshold = None)

    # for i, line in enumerate(input_lines):
    #     predicted.read(line)
    #     auc, optimal_f1_point = b.compare(predicted = predicted.oie,
    #                             matchingFunc = matchingFunc,
    #                             output_fn = out_filename,
    #                             error_file = None,
    #                             binary = False)
    #     fields = line.split('\t')
    #     fields[2] = str(optimal_f1_point[0]) # Replace confidence with the precision of the extraction
    #     # fields[2] = str(optimal_f1_point[3]) # Replace confidence with the optimal confidence of the extraction
    #     out_f.write('\t'.join(fields)+'\n')

    lines = []
    all_confidences = []
    for i, line in enumerate(input_lines):
        sentence, extraction, confidence = line.split('\t')
        if i == 0:
            old_sentence = sentence
        if i == len(input_lines)-1:
            old_sentence = ''
        if sentence != old_sentence:
            predicted.read(''.join(lines))
            auc, optimal_f1_point, _ = b.compare(predicted = predicted.oie,
                                    matchingFunc = matchingFunc,
                                    output_fn = out_filename,
                                    error_file = None,
                                    binary = False)
            optimal_confidence = optimal_f1_point[3]
            all_confidences.append(optimal_confidence)

            # Predict 'k' number of extractions above the threshold
            # all_confs = []
            # for ext in lines:
            #     fields = ext.split('\t')
            #     all_confs.append(float(fields[2]))
            # all_confs = sorted(all_confs)
            # correct_ratio = 0
            # if optimal_confidence != 0:
            #     correct_ratio = all_confs.index(optimal_confidence)

            # # Predict the optimal prob / max prob ratio
            # max_confidence = max([float(ext.split('\t')[2]) for ext in lines])
            # correct_ratio = math.exp(optimal_confidence)/math.exp(max_confidence) if optimal_confidence != 0 else 1.0

            for ext in lines:
                fields = ext.split('\t')
                fields[2] = str(optimal_confidence)
                # confidence = float(fields[2])
                # if confidence > optimal_confidence:
                #     fields[2] = '1'
                # else:
                #     fields[2] = '0'
                out_f.write('\t'.join(fields)+'\n')
            lines = []
            old_sentence = sentence
        lines.append(line)

    all_confidences = np.array(all_confidences)
    # print(np.sort(all_confidences))
    # print('Average: ', np.mean(all_confidences), np.std(all_confidences))
    out_f.close()
    return 

def main():
    global args
    parser = parse_args()
    args = parser.parse_args()

    run_carb(args.inp_fp, args.out_fp, args.gold_fp)

if __name__ == '__main__':
    main()

