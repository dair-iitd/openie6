import os
import sys
import ipdb
import spacy
import argparse
from multiset import Multiset

parser = argparse.ArgumentParser()
parser.add_argument('--inp_fp',type=str)
parser.add_argument('--inp_type',type=str) # gold, allennlp

args = parser.parse_args()
f = open(args.inp_fp, 'r')

nlp = spacy.load("en_core_web_sm")
sentenceD = dict()
if args.inp_type == 'gold':
    for line in f:
        line = line.strip('\n')
        fields = line.split('\t')
        sentence = fields[0]
        rel, args = fields[1], ' '.join(fields[2:])
        if sentence not in sentenceD:
            sentenceD[sentence] = list()
        sentenceD[sentence].append([rel, args])
elif args.inp_type == 'allennlp':
    for line in f:
        line = line.strip('\n')
        fields = line.split('\t')
        sentence = fields[0]
        try:
            arg1 = line[line.index('<arg1>') + 6:line.index('</arg1>')]
        except:
            arg1 = ""
        try:
            rel = line[line.index('<rel>') + 5:line.index('</rel>')]
        except:
            rel = ""
        try:
            arg2 = line[line.index('<arg2>') + 6:line.index('</arg2>')]
        except:
            arg2 = ""

        if sentence not in sentenceD:
            sentenceD[sentence] = list()
        sentenceD[sentence].append([rel, arg1+' '+arg2])

light_verbs = ["take", "have", "give", "do", "make"]+["has","have","be","is","were","are","was","had","being","began","am","following","having","do","does",
"did","started","been","became","left","help","helped","get","keep","think","got","gets","include",
"suggest","used","see","consider","means","try","start","included","lets","say","continued",
"go","includes","becomes","begins","keeps","begin","starts","said"]+["stop", "begin", "start", "continue", "say"]

def pos_tags(spacy_sentence):
    pos, pos_indices, pos_words = [], [], []
    for token_index, token in enumerate(spacy_sentence):
        if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
            pos.append(1)
            pos_indices.append(token_index)
            pos_words.append(token.lower_)
        else:
            pos.append(0)
    pos.append(0); pos.append(0); pos.append(0)
    return pos, pos_indices, pos_words

def verb_tags(spacy_sentence):
    verb, verb_indices, verb_words = [], [], []
    for token_index, token in enumerate(spacy_sentence):
        if token.pos_ in ['VERB'] and token.lower_ not in light_verbs:
            verb.append(1)
            verb_indices.append(token_index)
            verb_words.append(token.lower_)
        else:
            verb.append(0)
    verb.append(0); verb.append(0); verb.append(0)
    return verb, verb_indices, verb_words

pos_violation, hvc_violation, hve_violation, hvd_violation = 0, 0, 0, 0
for sentence in sentenceD:
    nl_sentence = nlp(sentence)
    _, _, verbs = verb_tags(nl_sentence) 
    _, _, pos_words = pos_tags(nl_sentence) 
    all_extraction_words, all_rel_words = list(), list()
    head_verb_extractions = 0
    for extraction in sentenceD[sentence]:
        rel, args = extraction
        extraction_str = rel+' '+args
        rel_words = rel.lower().split()
        extraction_words = extraction_str.lower().split()
        all_extraction_words = all_extraction_words + extraction_words
        all_rel_words = all_rel_words+rel_words
        rel_intersection = set(rel_words).intersection(set(verbs))
        if len(rel_intersection) > 1:
            hve_violation += 1
        if len(rel_intersection) > 0:
            head_verb_extractions += 1
    if head_verb_extractions < len(verbs):
        hvd_violation += 1
    pos_violation += len(Multiset(pos_words) - Multiset(all_extraction_words))
    all_rel_verb_words_m = Multiset(all_rel_words).intersection(Multiset(verbs))
    hvc_violation += len(Multiset(verbs) - all_rel_verb_words_m) + len(all_rel_verb_words_m-Multiset(verbs))

print('POSC & HVC & HVE & HVD')
print(f'{pos_violation} & {hvc_violation} & {hve_violation} & {hvd_violation}')    
