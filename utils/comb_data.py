import os
import ipdb
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp')
    parser.add_argument('--out')
    parser.add_argument('--conj')

    return parser

parser = parse_args()
args = parser.parse_args()

conj_mapping = dict()
content = open(args.conj).read()
for example in content.split('\n\n'):
    for i, line in enumerate(example.strip('\n').split('\n')):
        if i == 0:
            orig_sentence = line
        else:
            conj_mapping[line] = orig_sentence

extr_mapping = dict()
for line in open(args.inp):
    line = line.strip('\n')
    sentence, extraction, confidence = line.split('\t')
    if sentence in conj_mapping:
        orig_sentence = conj_mapping[sentence]
        if orig_sentence not in extr_mapping:
            extr_mapping[orig_sentence] = dict()

