import ipdb
import pdb
import sys
import os
import regex as re
import argparse
import toolz
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.common.util import import_submodules
from distutils.util import strtobool

sys.path.insert(0,"code")

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--topk', type=int)

    return parser

def generate_probs(model_dir, inp_fp, weights_fp, topk, out_ext, cuda_device, overwrite, batch_size, extraction_ratio, out):
    import_submodules('imojie')

    args = argparse.Namespace()
    args.archive_file = model_dir
    args.cuda_device = cuda_device
    args.embedding_sources_mapping = {}
    args.extend_vocab = None
    args.batch_weight_key = ''
    args.output_file = ''
    args.overrides = "{'model': {'token_based_metric': null}, 'iterator': {'batch_size': "+str(batch_size)+", \
        'instances_per_epoch': null}, 'trainer':{'num_epochs':1}, 'dataset_reader': {'max_tokens': 10000, \
            'gradients': false, 'max_extractions': 30, 'extraction_ratio': "+str(extraction_ratio)+", 'probability': true \
                 }, 'validation_dataset_reader': null}"
    args.weights_file = weights_fp
    args.input_file = inp_fp
    probs = evaluate_from_args(args)

    probsD = dict()
    # For some reason the last batch results are repeated in the probs 
    # Not an issue as they are just overwritten while forming the probsD
    for i in range(len(probs['example_ids'])):
        probsD[probs['example_ids'][i]] = probs['probs'][i]
    lines = open(inp_fp).readlines()

    all_fields = []
    for line_number, line in enumerate(lines):
        line = line.strip('\n')
        fields = line.split('\t')
        if line_number not in probsD: # the example is too large and rejected by dataloader ('max_tokens' argument)
            continue
        # Removing appended extractions after reranking
        fields[0] = fields[0].split('[SEP]')[0].strip()
        fields[2] = str(probsD[line_number])
        fields.append(str(line_number))
        all_fields.append('\t'.join(fields))

    if topk == None:
        return all_fields
    else:
        # sorting all_fields according to the confidences assigned by bert_encoder
        all_fields_sorted = []
        prev_sent=None
        exts=[]
        for f in all_fields:
            sent = f.split('\t')[0]
            if sent!=prev_sent:
                if prev_sent!=None:
                    exts = toolz.unique(exts, key=lambda x: x.split('\t')[1])
                    exts = sorted(exts, reverse=True, key= lambda x: float(x.split('\t')[2]) )
                    if topk != None:
                        exts = exts[:topk]
                    all_fields_sorted.extend(exts)
                prev_sent=sent
                exts=[f]
            else:
                exts.append(f)
        exts = sorted(exts, reverse=True, key= lambda x: float(x.split('\t')[2]) )
        all_fields_sorted.extend(exts)
        open(out,'w').write('\n'.join(all_fields_sorted))
        print('Probabilities written to: ',out)

        return all_fields_sorted


def rescore(inp_fp, topk=None, out_ext=None, model_dir='imojie/models/be', cuda_device=0, overwrite=True, ext_ratio=1, batch_size = 64):
    weights_fp = model_dir + '/best.th'
    return generate_probs(model_dir, inp_fp, weights_fp, topk, out_ext, cuda_device, overwrite=overwrite, extraction_ratio=ext_ratio, batch_size=batch_size, out=None)


def main():
    parser = parse_args()
    args = parser.parse_args()
    generate_probs('../models/rescore_model', args.inp, '../models/rescore_model/best.th', args.topk, None, 0, overwrite=True, extraction_ratio=1, batch_size=256, out=args.out)

if __name__ == '__main__':
    main()