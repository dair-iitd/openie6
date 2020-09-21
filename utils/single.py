import sys
sys.path.insert(0, 'carb')

from carb import Benchmark
from oie_readers.extraction import Extraction
from matcher import Matcher

import ipdb
import os
import copy
import tqdm

benchmark = Benchmark('carb/data/gold/dev.tsv')
inp1 = sys.argv[1]
inp2 = sys.argv[2]

def get_predictions(fp):
    old_sentence = ''
    all_predictions = dict()
    for line in open(fp,'r'):
        line = line.strip('\n')
        sentence, extraction, confidence = line.split('\t')
        if old_sentence == '':
            old_sentence = sentence
            all_predictions[sentence] = []
        if old_sentence != sentence:
            all_predictions[sentence] = []
            old_sentence = sentence
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

        if not ( arg1 or arg2 or rel):
            continue
        curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = sentence, confidence = float(confidence))
        curExtraction.addArg(arg1)
        curExtraction.addArg(arg2)        
        all_predictions[sentence].append(curExtraction)

    return all_predictions

def get_scores(predictions, benchmark):
    scoresD = dict()
    copy_benchmark = copy.deepcopy(benchmark)    
    copy_benchmark.gold = None

    for sentence in tqdm.tqdm(predictions):
        if sentence not in benchmark.gold:
            continue
        gold_exts = benchmark.gold[sentence] 
        copy_benchmark.gold = {sentence: gold_exts}
        auc, optimal_f1_point, last_f1_point = copy_benchmark.compare(predicted={sentence: predictions[sentence]}, matchingFunc=Matcher.binary_linient_tuple_match, output_fn='/dev/null', error_file=None, binary=False)
        scoresD[sentence] = last_f1_point[:-1]
    
    return scoresD

def max_diff(scores1, scores2):
    diffL, diffD = [], {}
    for sentence in scores1:
        # assert sentence in scores2
        # if sentence not in scores2:
        #     ipdb.set_trace()
        #     continue
        diffL.append((sentence, scores1[sentence], scores2[sentence]))
        diffD[sentence] = [scores1[sentence], scores2[sentence]]
    diffL = sorted(diffL, key=lambda x: x[1][2]-x[2][2], reverse=True)
    return diffL, diffD

def extractions_to_str(extractions):
    str_set = set()
    for extr in extractions:
        str_extr = ' '.join(extr.args) + ' ' + extr.pred
        str_set.add(str_extr)
    return str_set

def normalize_confidence(diffD, predictions1, predictions2):
    np1, np2 = dict(), dict()
    for sentence, extractions1 in predictions1.items():
        if sentence not in diffD:
            continue
        extractions2 = predictions2[sentence]
        extr1_set = extractions_to_str(extractions1)
        extr2_set = extractions_to_str(extractions2)
        if extr1_set == extr2_set:
            score1 = diffD[sentence][0][2]
            score2 = diffD[sentence][1][2]
            if score2 > score1:
                np2[sentence] = extractions2
                np1[sentence] = extractions2
            else:
                np2[sentence] = extractions1
                np1[sentence] = extractions1
        else:
            np1[sentence] = extractions1
            np2[sentence] = extractions2
    return np1, np2

print(inp1)
predictions1 = get_predictions(inp1)
auc, optimal_f1_point, last_f1_point = benchmark.compare(predicted=predictions1,
                matchingFunc=Matcher.binary_linient_tuple_match,
                output_fn='/dev/null', error_file=None,
                binary=False)
print({'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_lastf1': last_f1_point[2]})

print(inp2)
predictions2 = get_predictions(inp2)
auc, optimal_f1_point, last_f1_point = benchmark.compare(predicted=predictions2,
                matchingFunc=Matcher.binary_linient_tuple_match,
                output_fn='/dev/null', error_file=None,
                binary=False)
print({'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_lastf1': last_f1_point[2]})

scores1 = get_scores(predictions1, benchmark)
scores2 = get_scores(predictions2, benchmark)
diffL, diffD = max_diff(scores1, scores2)
# np1, np2 = normalize_confidence(diffD, predictions1, predictions2)

# print('np1')
# auc, optimal_f1_point, last_f1_point = benchmark.compare(predicted=np1,
#                 matchingFunc=Matcher.binary_linient_tuple_match,
#                 output_fn='/dev/null', error_file=None,
#                 binary=False)
# print({'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_lastf1': last_f1_point[2]})
# print('np2')
# auc, optimal_f1_point, last_f1_point = benchmark.compare(predicted=np2,
#                 matchingFunc=Matcher.binary_linient_tuple_match,
#                 output_fn='/dev/null', error_file=None,
#                 binary=False)
# print({'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_lastf1': last_f1_point[2]})

# ipdb.set_trace()
print('\n'.join([elem[0]+'\t'+str(elem[1])+'\t'+str(elem[2]) for elem in diffL]))
