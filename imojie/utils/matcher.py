# This script matches sentences among 2 input files. It can take the sentence file as well as the extractions file as input (test.txt, dev.txt or test.tsv, dev.tsv )
# python sentence_matcher.py --inp_1 <file1> --inp_2 <file2> --threshold_1 0 --threshold_2 0 --type sentences
# if type is extractions, it also outputs the difference in the extractions of the two systems

import sys
import argparse
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_1', type=str, required=True)
    parser.add_argument('--inp_2', type=str, required=True)
    parser.add_argument('--threshold_1', type=float)
    parser.add_argument('--threshold_2', type=float)
    parser.add_argument('--type', type=str, required=True) # sentences, extractions

    return parser


def getsent(filename):
    lines = open(filename,'r').read().strip().split('\n')
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split('\t')[0].strip()

    return set(lines)

def get_order_extractions(filename, threshold):
    lines = open(filename,'r').read().strip().split('\n')
    set_lines, example, old_sentence = set(), '', ''
    for i in range(len(lines)):
        fields = lines[i].strip().split('\t')
        score = float(fields[2])
        if threshold != None and score < threshold:
            continue
        sentence = fields[0].strip()
        extraction = fields[1].strip()
        confidence = fields[2].strip()
        if sentence != old_sentence:
            if i!=0:
                set_lines.add(example)
            example = f'{sentence}\t{extraction}'
            old_sentence = sentence
        else:
            example = f'{example}\t{extraction}'
        # set_lines.add(sentence + '\t' + extraction)
        # set_lines.add(sentence + '\t' + extraction + '\t' + confidence)
    return set_lines

def getextractions(filename, threshold):
    lines = open(filename,'r').read().strip().split('\n')
    set_lines = set()
    for i in range(len(lines)):
        fields = lines[i].strip().split('\t')
        score = float(fields[2])
        if threshold != None and score < threshold:
            continue
        sentence = fields[0].strip()
        extraction = fields[1].strip()
        confidence = fields[2].strip()
        set_lines.add(sentence + '\t' + extraction)
        # set_lines.add(sentence + '\t' + extraction + '\t' + confidence)

    return set_lines


# returns extractions for a particular sentence
def extractions_for_sentence(sentence, extraction_set):
    ext=[]
    for e in extraction_set:
        if e.split('\t')[0]==sentence:
            ext.append('\t'.join(e.split('\t')[1:]))
    return ext

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    if args.type == 'sentences':
        s1 = getsent(args.inp_1)
        s2 = getsent(args.inp_2)
    elif args.type  == 'extractions':
        s1 = getextractions(args.inp_1, args.threshold_1)
        s2 = getextractions(args.inp_2, args.threshold_2)
    elif args.type  == 'order_extractions':
        s1 = get_order_extractions(args.inp_1, args.threshold_1)
        s2 = get_order_extractions(args.inp_2, args.threshold_2)

    if(s1==s2):
        print("matched")
    else:
        print("file1 has extra")
        print(len(s1-s2))
        print("file2 has extra")
        print(len(s2-s1))

        sents = [x.split('\t')[0] for x in list(s1-s2)]
        common_ext = s1 - (s1-s2)
        for sent in set(sents):
            print(sent)
            print("========= COMMON\n",'\n'.join(extractions_for_sentence(sent, common_ext)))
            print("========= SYS1\n",'\n'.join(extractions_for_sentence(sent, s1-s2)))
            print("========= SYS2\n",'\n'.join(extractions_for_sentence(sent, s2-s1)))
            print("==================================")

