import os
import regex as re
import random
import ipdb
import argparse
import json
from distutils import util
import copy

# import noie
# from code import noie
from imojie import bert_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--pred_dir', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--type', type=str, default='single')
    parser.add_argument('--bert', type=lambda x:bool(util.strtobool(x)), default=False)

    parser.add_argument('--inp_fp', type=str)
    parser.add_argument('--out_fp', type=str)

    return parser

def bert_replace(seq_tokens):
    new_seq_tokens = []
    for st in seq_tokens:
        new_st = []
        for t in st:
            if t in bert_utils.unk_mapping: 
                new_st.append(bert_utils.unk_mapping[t])
            else:
                if(t[:2] == '##'):
                    if len(new_st) != 0:
                        new_st[-1] = new_st[-1] + t[2:]
                    else:
                        new_st.append(t[2:]) # Will never be correct but can happend
                else:
                    new_st.append(t)
        new_seq_tokens.append(new_st)
    return new_seq_tokens

def clean(extraction):
    # Ensure single <arg1>... tags
    if extraction.count('<arg1>') != 1:
        return False
    if extraction.count('</arg1>') != 1:
        return False
    if extraction.count('<arg2>') != 1:
        return False
    if extraction.count('</arg2>') != 1:
        return False
    if extraction.count('<rel>') != 1:
        return False
    if extraction.count('</rel>') != 1:
        return False

    return True

def process_append(input_lines, test_sents):
    ## both inp_fp and out_fp contains the directory
    out_lines = []
    rerank_lines = []
    append_lines = []
    
    for i, line in enumerate(input_lines):
        jline = json.loads(line)
        seq_tokens = jline['predicted_tokens'][0] # Consider only the best beam
        seq_probs = jline['predicted_log_probs'] # Contains score of only the best beam

        seq_tokens = bert_replace([seq_tokens])[0]

        extraction_scores, extractions, extraction = [], [], ''
        extraction_num = 0

        for token_index, token in enumerate(seq_tokens):
            if(token == '[SEP]' and len(extraction.split()) != 0):
                # extraction_scores.append(seq_probs[extraction_num] / len(extraction.split()))
                try:
                    extraction_scores.append(seq_probs[extraction_num])
                except:
                    ipdb.set_trace()
                extraction_num += 1
                extractions.append(extraction.strip())
                extraction = ''
                continue
            extraction += token + ' '

        test_sent = test_sents[i].strip('\n')
        dedup_extractions, dedup_scores = [], []
        for extraction_num in range(len(extractions)):
            extraction = extractions[extraction_num]
            score = extraction_scores[extraction_num]
            if extraction not in dedup_extractions:
                dedup_extractions.append(extraction)
                dedup_scores.append(score)

        extractions, extraction_scores = dedup_extractions, dedup_scores

        for extraction_num, extraction in enumerate(extractions):
            if not clean(extraction):
                continue
            out_line = test_sent + '\t' + extraction + '\t' + str(extraction_scores[extraction_num])
            out_lines.append(out_line)

        sorted_extractions = sorted(zip(extractions, extraction_scores), reverse=True, key=lambda x: x[1])
        sorted_extractions_scores = [se[1] for se in sorted_extractions]
        sorted_extractions = [se[0] for se in sorted_extractions]

        for e, extraction in enumerate(sorted_extractions):
            other_extractions = sorted_extractions[:e] + sorted_extractions[e+1:]
            rerank_line = test_sent + ' [SEP] [CLS] ' + ' [SEP] [CLS] '.join(other_extractions) + '\t' + extraction + '\t' + str(sorted_extractions_scores[e])
            rerank_lines.append(rerank_line)

        append_line = test_sent + '\t' + '\t'.join(sorted_extractions) + '\t' + ','.join([str(f) for f in sorted_extractions_scores])
        append_lines.append(append_line)
    return '\n'.join(out_lines), '\n'.join(rerank_lines), '\n'.join(append_lines)


def process_coverage(input_lines, test_sents, bert=False):
    ## both inp_fp and out_fp contains the directory
    out_lines = []
    
    for i, line in enumerate(input_lines):
        jline = json.loads(line)
        seq_tokens = jline['predicted_tokens'][0] # Consider only the best beam
        token_scores = jline['predicted_log_probs'][0] # Contains individual word scores of only the best beam

        # ipdb.set_trace()
        seq_tokens = bert_replace([seq_tokens])[0]

        extraction_scores = []
        extractions = []
        extraction_score = None

        for token_index, token in enumerate(seq_tokens):
            if(token == '</arg2>'):
                extraction_score.append(token_scores[token_index])
                extraction_score_avg = sum(extraction_score) / float(len(extraction_score))
                # extraction_score_avg = sum(extraction_score) 
                extraction_scores.append(extraction_score_avg)
                extractions.append(extraction + ' ' + token)
                continue
            if(token == '<arg1>'):
                extraction_score = [token_scores[token_index]]
                extraction = token
                continue
            if(type(extraction_score) != type(None)):
                extraction_score.append(token_scores[token_index])
                extraction += ' ' + token

        test_sent = test_sents[i].strip('\n')
        for extraction_num, extraction in enumerate(extractions):
            out_line = test_sent + '\t' + extraction + '\t' + str(extraction_scores[extraction_num])
            out_lines.append(out_line)
    
    return '\n'.join(out_lines)


def process_single(input_lines, test_sents, threshold):
    ## in_fp contains the directory
    ## out_fp does not contain the directory
    out_lines_1 = []
    out_lines_5 = []
    for i, line in enumerate(input_lines):
        try:
            jline = json.loads(line)
        except:
            continue # Can happen when model produces double quotes within the output TODO: Escape it
        if 'predicted_log_probs' in jline:
            scores = jline['predicted_log_probs']
        elif 'class_log_probabilities' in jline:
            scores = jline['class_log_probabilities']
        else:
            raise Exception("Confidence is not recieved")

        seq_tokens = jline['predicted_tokens']

        seq_tokens = [" ".join(st) for st in seq_tokens]
        seq_tokens = [re.sub(r'\[\ unused\ ##(\d+)\ \]',r'[unused\1]', st) for st in seq_tokens]
        seq_tokens = [st.split(' ') for st in seq_tokens]
        seq_tokens = bert_replace(seq_tokens)

        test_sent = test_sents[i].strip('\n')
        out_line = ""
        for j in range(len(scores)):
            if(type(threshold) != type(None) and scores[j] < threshold):
                continue
            extraction = " ".join(seq_tokens[j])
            if not clean(extraction):
                continue
            out_line = test_sent + "\t" + extraction + "\t" + str(scores[j])
            out_lines_5.append(out_line)
            if(j == 0):
                out_lines_1.append(out_line)

    return "\n".join(out_lines_5)

def main():
    parser = parse_args()
    args = parser.parse_args()

    pred_dir = args.pred_dir
    test_sents = open(args.test).readlines()

    if args.inp_fp != None:
        output_lines, _, _ = process_append(open(args.inp_fp,'r').readlines(), test_sents)
        open(args.out_fp,'w').write(output_lines)
        return

    if(args.type == 'single'):
        # Entire set of predictions
        for fil in os.listdir(pred_dir):
            if('.jsonl' not in fil):
                continue
            # process_single(exp_dir, pred_dir+'/'+fil, 'pro_'+fil.split('.jsonl')[0]+'.txt', test_sents, args.threshold)
            input_file = pred_dir+'/'+fil
            print(input_file)
            output_lines = process_single(open(input_file,'r').readlines(), test_sents, args.threshold)
            open(pred_dir+'/pro_'+fil.split('.jsonl')[0]+'.txt','w').write(output_lines)
    elif(args.type == 'coverage'):
        for fil in os.listdir(pred_dir):
            if('.jsonl' not in fil):
                continue
            input_file = pred_dir+'/'+fil
            print(input_file)
            output_lines = process_coverage( open(input_file,'r').readlines(), test_sents)
            open( pred_dir+'/pro_'+fil.split('.jsonl')[0]+'.txt', 'w').write(output_lines)

    elif(args.type == 'append'):
        for fil in os.listdir(pred_dir):
            if('.jsonl' not in fil):
                continue
            input_file = pred_dir+'/'+fil
            print(input_file)
            output_lines, rerank_lines, append_lines = process_append(open(input_file,'r').readlines(), test_sents)
            open(pred_dir+'/pro_'+fil.split('.jsonl')[0]+'.txt','w').write(output_lines)
            open(pred_dir+'/pro_'+fil.split('.jsonl')[0]+'.txt.rerank','w').write(rerank_lines)
            # open(pred_dir+'/pro_'+fil.split('.jsonl')[0]+'.txt.append','w').write(append_lines)


if __name__ == '__main__':
    main()    


