import os
import ipdb
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fp', type=str)
    parser.add_argument('--out_fp', type=str)

    return parser

parser = parse_args()
args = parser.parse_args()

out_f = open(args.out_fp, 'w')
exampleD = dict()

for i, line in enumerate(open(args.inp_fp)):
    line = line.strip('\n')
    if '[unused' in line: # starting new example
        if i != 0:
            exampleD[sentence] = labels
        labels = []
        sentence = line
    else:
        labels.append(line)        

for sentence, labels in exampleD.items():
    sentence_np = np.array(sentence.split())
    extractions, len_extractions = [], []
    for label in labels:
        label = np.array(label.split()[:-3])
        arg1 = ' '.join(sentence_np[np.where(label == 'ARG1')])
        arg2 = ' '.join(sentence_np[np.where(label == 'ARG2')])
        rel = ' '.join(sentence_np[np.where(label == 'REL')])
        extraction = arg1 + ' ' + rel + ' ' + arg2
        len_extraction = len(extraction.split())
        extractions.append(extraction)
        len_extractions.append(len_extraction)

    for i, label in enumerate(labels):
        other_exts = extractions[:i] + extractions[i+1:]
        len_other_exts = sum(len_extractions[:i]) + sum(len_extractions[i+1:])
        input = sentence + ' ' + ' '.join(other_exts)
        output = label + ' ' + ' '.join(['NONE']*len_other_exts)
        out_f.write(input+'\n'+output+'\n')
