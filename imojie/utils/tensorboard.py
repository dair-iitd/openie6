import argparse
import os
import ipdb

from torch.utils.tensorboard import SummaryWriter

conjunctions = ['and', 'but', 'or']

parser = argparse.ArgumentParser()
parser.add_argument('--inp_fp', type=str)
args = parser.parse_args()

extractionD, sentence_idx, idxD = dict(), 0, dict()
old_sentence = ''
for line in open(args.inp_fp):
    line = line.strip('\n')
    sentence, extraction, confidence = line.split('\t')
    if sentence != old_sentence:
        extractionD[sentence] = []
        idxD[sentence] = sentence_idx
        sentence_idx += 1
        old_sentence = sentence
    extractionD[sentence].append([extraction, confidence])

writer = SummaryWriter(args.inp_fp+'.logs')
for sentence in extractionD:
    sentence_idx = idxD[sentence]
    conj = False
    for conjunction in conjunctions:
        if conjunction in sentence.split():
            conj = True
    extractions_str = ''
    for (extraction, confidence) in extractionD[sentence]:
        extractions_str = f'{extractions_str}  \n{float(confidence):0.3f}: {extraction}'
    # final_str=f'{sentence}\n\nPredictions:\n{extractions_str}'
    final_str=f'Predictions:\n{extractions_str}'
    writer.add_text(str(sentence_idx)+' '+sentence, final_str, 0)
writer.close()
