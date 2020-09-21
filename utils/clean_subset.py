import os
import ipdb
import sys
import argparse

def load_conj_mapping(conj_fp):
    conj_mapping = dict()
    conj_mapping_values = set()
    content = open(conj_fp).read()
    for example in content.split('\n\n'):
        for i, line in enumerate(example.strip('\n').split('\n')):
            if i == 0:
                orig_sentence = line
            else:
                conj_mapping[line] = orig_sentence
    conj_mapping_values = conj_mapping.values()
    return conj_mapping

def main(args):
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_clean')
    parser.add_argument('--inp_orig')
    parser.add_argument('--inp_conj')
    parser.add_argument('--out')

    args = parser.parse_args()
    main(args)