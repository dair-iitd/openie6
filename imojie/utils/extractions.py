import os
import sys
import ipdb
import regex as re
import random
from tqdm import tqdm
from distutils.util import strtobool
import argparse
import numpy as np
import pickle

random.seed(1234)
global args

## Utility for generating extractions in different formats
## inp_fp: data/train/wiki5/extractions.txt - the original extractions
## out_fp: <output_file>
## threshold: extractions above this threshold will be considered
## delimitters: whether we need to use delimitters - <arg1>, </arg1>, etc
## ext_type: single, multiple - how many have to be written to a single line

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext_type', type=str, default='single')
    parser.add_argument('--model', action='append', type=str)
    parser.add_argument('--out_fp', action='append', type=str)
    parser.add_argument('--eval', type=str, action='append')
    parser.add_argument('--sentences_fp', type=str, default='sentences.txt')
    parser.add_argument('--extractions_fp', type=str)
    parser.add_argument('--threshold', type=float) # Required for openie4, openie5, not for clausie
    parser.add_argument('--rand', default=False, type=lambda x:bool(strtobool(x)))

    return parser

def get_extraction(line, orig_sent):
    # Seperate the confidence
    line_split = line.split()
    confidence = float(line_split[0])
    extraction = " ".join(line_split[1:])
    # Remove the context tag
    if 'Context(' in line:
        extraction = " ".join(extraction.split('):')[1:])
    # Remove the brackets from the inserted [is], [of], [from], etc
    for bracket in re.findall(r'(\[(.*?)\])', extraction):
        if bracket[0] in orig_sent:
            continue
        extraction = extraction.replace(bracket[0], bracket[1])
    # Clean the L:, T: tags
    extraction = extraction.replace('L:','')
    extraction = extraction.replace('T:','')
    fields = extraction[1:-1].split(';') # Removes '(', ')'
    
    if len(fields) < 3: # Happens for wrong sentences, will be filtered later on, when check with orig sentences
        return extraction
    arg1, rel, arg2 = fields[0], fields[1], " ".join(fields[2:])
    extraction = '<arg1> '+arg1+' </arg1> <rel> '+rel+ ' </rel> <arg2> '+arg2+' </arg2>'

    return confidence, extraction

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def check_extraction(line):
    ## Check if the line is an extraction
    first_word = line.split()[0]
    if is_number(first_word) and ';' in line:
        return True
    elif "Context(" in line:
        return True
    elif "List(" in line:
        return True
    return False

def orig_sentences(inp_fp):
    orig_sentences = []
    for line in open(inp_fp):
        line = line.strip('\n')
        orig_sentences.append(line)    
    return set(orig_sentences)

def get_extractions(inp_fp, orig_sents, ext_type):
    new_sentence = True
    sents_covered = dict()
    extD = dict()

    for sent in orig_sents:
        sents_covered[sent] = 0
    for i, line in tqdm(enumerate(open(inp_fp))):
        line = line.strip('\n')
        if line == "Panic on line: helped gain helped":
            continue

        if line == "":
            new_sentence = True
            if sentence not in orig_sents or sents_covered[sentence] == 1:
                continue
            sents_covered[sentence] = 1    
            if check_extraction(sentence):
                continue
            extD[sentence] = extractions
            continue

        if new_sentence:
            sentence = line
            extractions = list()
            new_sentence = False
            continue
        extraction = get_extraction(line, sentence)
        extractions.append(extraction)

    return extD

def arg_subset(e1, e2):
    e1_arg = re.search('<arg2>(.*)</arg2>', e1).group(1)
    e2_arg = re.search('<arg2>(.*)</arg2>', e2).group(1)
    if e1_arg.strip() == '':
        return False
    if e1_arg in e2_arg:
        return True
    else:
        return False

def write_extractions(extD, out_fp, ext_type, threshold):
    out_f = open(out_fp,'w')

    num_extractions = 0
    for extraction_index, (sentence, extractions) in enumerate(tqdm(extD.items())):
        # Filter extractions based on threshold
        filter_extractions = []
        for ext in extractions:
            if threshold != None and ext[0] < threshold:
                continue
            num_extractions += 1
            filter_extractions.append(ext) 
        extractions = filter_extractions

        dedup_extractions = [(e[0],' '.join(e[1].split())) for e in extractions] # Remove extra spaces
        seen = set()
        extractions = [(a,b) for a,b in dedup_extractions if not (b in seen or seen.add(b))]

        # uncomment only for clausie
        # sub_extractions = []
        # for ext_i, extraction in enumerate(extractions):
        #     sub = False
        #     for ext_j, other_extraction in enumerate(extractions):
        #         if ext_i == ext_j:
        #             continue
        #         if arg_subset(extraction[1], other_extraction[1]) == True:
        #             sub = True
        #     if not sub: 
        #         sub_extractions.append(extraction)
        # extractions = sub_extractions


        if ext_type=="single":
            # for beam search style models
            for ext in extractions:
                out_f.write(sentence+"\t"+ext[1]+'\t'+str(ext[0])+"\n")
        
        elif ext_type=="multiple":
            # for bert append models
            exts = [e[1] for e in extractions]
            scores = [str(e[0]) for e in extractions]
            out_f.write(sentence+"\t"+"\t".join(exts)+'\t'+','.join(scores)+"\n")

        elif ext_type == "rerank":
            extractions = sorted(extractions, reverse=True, key=lambda x: x[0])
            for e, extraction in enumerate(extractions):
                other_exts = extractions[:e] + extractions[e+1:]
                other_exts = [oe[1] for oe in other_exts]
                other_sentence = sentence + ' [SEP] [CLS] ' + ' [SEP] [CLS] '.join(other_exts)
                out_f.write(other_sentence+'\t'+extraction[1]+'\t'+str(extraction[0])+'\n')

        elif ext_type =="append":
            # extractions = sorted(extractions, reverse=True, key=lambda x: x[0])
            for e, extraction in enumerate(extractions):
                previous_exts = extractions[:e]
                previous_exts = [oe[1] for oe in previous_exts]
                if len(previous_exts) > 0:
                    source = sentence + ' [SEP] [CLS] ' + ' [SEP] [CLS] '.join(previous_exts)
                else:
                    source = sentence
                out_f.write(source +'\t'+extraction[1]+'\t'+str(extraction[0])+'\n')

        elif ext_type=="concat":
            # for bert append models
            exts = [e[1] for e in extractions]
            scores = [str(e[0]) for e in extractions]
            out_f.write(sentence+"\t"+" ".join(exts)+'\t'+','.join(scores)+"\n")

    out_f.close()
    return

def comb_pkls(inps, rand):
    out = dict()
    if rand == False:
        for inp_i, inp in enumerate(inps):
            for key in inp:
                if key in out:
                    out[key].extend(inp[key])
                else:
                    out[key] = inp[key]
    else:
        all_keys = list()
        for inp in inps:
            all_keys.extend(inp.keys())
        all_keys = list(dict.fromkeys(all_keys).keys())
        for key_i, key in enumerate(all_keys):
            system_id = random.randrange(0,len(inps))
            chosen_system = inps[system_id]
            value = None
            if key not in chosen_system:
                if key in inps[0]:
                    value = inps[0][key]
            else:
                value = chosen_system[key]
            
            if value != None:
                out[key] = value

    return out

def main():
    global args

    parser = parse_args()
    args = parser.parse_args()

    for eval_i, eval_ in enumerate(args.eval):
        model_nameD = {'clausie':'c', 'oie4':'4', 'oie5':'5', 'rnnoie':'r', 'minie':'m', 'carb':'c', 'senseoie':'s'}
        sentences_fp = 'data/{}/{}'.format(eval_, args.sentences_fp)
        comb_str = ''

        if args.extractions_fp != None:
            inp_fps = [args.extractions_fp]
        else:
            inp_fps = [0]*len(args.model)
            for i, model in enumerate(args.model):
                inp_fps[i] = 'data/{}/{}/extractions.txt'.format(eval_, model)
        
        if args.out_fp == None:
            for i, model in enumerate(args.model):
                comb_str += model_nameD[model]
            if len(inp_fps) == 1:
                comb_str = model
            else:
                if args.rand:
                    comb_str = 'comb_rand_'+''.join(sorted(comb_str))
                else:
                    comb_str = 'comb_'+''.join(sorted(comb_str))
            out_dir = 'data/{}/{}/'.format(eval_, comb_str)
            os.makedirs(out_dir, exist_ok=True)
            out_fp = out_dir+'/extractions.tsv'
        else:
            out_fp = args.out_fp[eval_i]
            
        sentences = orig_sentences(sentences_fp) 
        
        print("generating extractions ...")
        extDs = []
        for inp_fp in inp_fps:
            extD = get_extractions(inp_fp, sentences, args.ext_type)
            extDs.append(extD)

        extD = comb_pkls(extDs, rand=args.rand)

        print("writing extractions to file: ", out_fp)
        write_extractions(extD, out_fp, args.ext_type, args.threshold)

main()
