import sys

import os
import json
import ipdb
import ast
import argparse
import _jsonnet
import shutil
from distutils.util import strtobool
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_path', type=str) 
    parser.add_argument('--s', type=str) 
    parser.add_argument('--force', default=False, type=lambda x:bool(strtobool(x)))
    parser.add_argument('--recover', default=False, type=lambda x:bool(strtobool(x)))
    parser.add_argument('--debug', default=False, type=lambda x:bool(strtobool(x)))
    parser.add_argument('--overrides', default="", type=str)
    parser.add_argument('--test_fp', default="test/carb", type=str)
    parser.add_argument('--mode', default="train_test", type=str)
    parser.add_argument('--type', default="append", type=str) # append, single
    parser.add_argument('--perform', default="gen_pro_carb_compile", type=str)
    parser.add_argument('--num_process', default=2, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=float)
    parser.add_argument('--data', type=str)
    parser.add_argument('--order_sentences', type=str, default='')
    parser.add_argument('--decoder_type', type=str, default='')
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('--max_extractions', type=int)
    parser.add_argument('--max_tokens', type=int)
    parser.add_argument('--gpus', default=1, type=int)

    # grid-search parameters    
    parser.add_argument('--batch_size', type=int) 
    parser.add_argument('--samples_per_batch', type=int) 
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--target_embedding_dim', type=int)
    parser.add_argument('--encoder_lr', type=float)
    parser.add_argument('--decoder_lr', type=float)

    return parser

def main():
    import_submodules('imojie')

    parser = parse_args()
    args = parser.parse_args()
    
    model_dir = args.s
    if args.recover or args.mode == 'test':
        args.param_path = model_dir+'/config.json'

    args.overrides = ''
    configD = json.loads(_jsonnet.evaluate_snippet("", open(args.param_path, 'r').read()))
    overrideD = dict()
    overrideD['iterator'] = dict()
    overrideD['trainer'] = dict()
    overrideD['model'] = dict()
    overrideD['dataset_reader'] = dict()
    overrideD['validation_dataset_reader'] = dict()
    overrideD['trainer']['cuda_device'] = args.cuda_device

    if args.samples_per_batch != None:
        overrideD['iterator']['maximum_samples_per_batch'] = ["num_tokens", args.samples_per_batch]
    
    if args.max_extractions != None:
        overrideD['dataset_reader']['max_extractions'] = args.max_extractions
        overrideD['model']['max_extractions'] = args.max_extractions
    if args.max_tokens != None:
        overrideD['dataset_reader']['max_tokens'] = args.max_tokens
    
    if args.order_sentences != '':
        overrideD['dataset_reader']['order_sentences'] = args.order_sentences
        overrideD['validation_dataset_reader']['order_sentences'] = args.order_sentences

    if args.epochs != None:
        overrideD['trainer']['num_epochs'] = args.epochs

    ## grid search params
    if args.batch_size != None:
        overrideD['iterator']['batch_size'] = args.batch_size
    if args.decoder_layers != None:
        overrideD['model']['decoder_layers'] = args.decoder_layers
    if args.hidden_dim != None:
        overrideD['model']['encoder'] = dict()
        overrideD['model']['encoder']['feedforward'] = dict()
        overrideD['model']['encoder']['feedforward']['hidden_dims'] = args.hidden_dim
        overrideD['model']['attention'] = dict()
        overrideD['model']['attention']['tensor_1_dim'] = args.hidden_dim
        overrideD['model']['attention']['tensor_2_dim'] = args.hidden_dim
    if args.target_embedding_dim != None:
        overrideD['model']['target_embedding_dim'] = args.target_embedding_dim

    if 'parameter_groups' in configD['trainer']['optimizer']:
        param_group = configD['trainer']['optimizer']['parameter_groups']
    if args.encoder_lr != None:
        param_group[0][1]['lr'] = args.bert_lr
        overrideD['trainer']['optimizer'] = dict()
        overrideD['trainer']['optimizer']['parameter_groups'] = param_group
    if args.decoder_lr != None:
        param_group[1][1]['lr'] = args.decoder_lr
        overrideD['trainer']['optimizer'] = dict()
        overrideD['trainer']['optimizer']["lr"] = args.decoder_lr
        overrideD['trainer']['optimizer']['parameter_groups'] = param_group
    if args.debug:
        overrideD['iterator']['instances_per_epoch'] = str(100)
        overrideD['train_data_path'] = 'data/debug.tsv'

    if args.seed != None:
        overrideD['random_seed'] = args.seed

    if args.gpus > 1:
        overrideD['trainer']['cuda_device'] = list(range(args.gpus))
        overrideD['iterator']['batch_size'] = configD['iterator']['batch_size'] / args.gpus

    train_path = configD['train_data_path']
    if args.data != None: 
        train_data_path = args.data
        overrideD['train_data_path'] = train_data_path
        
    args.overrides = str(overrideD)

    # grid-search related
    if args.batch_size != None:
        overrideD['batch_size'] = args.batch_size
    
    if 'train' in args.mode:
        train_model_from_file( parameter_filename=args.param_path,
                                    serialization_dir=model_dir,
                                    force=args.force,
                                    recover=args.recover,
                                    overrides=args.overrides
                                )
    
    if 'test' in args.mode:
        append = False
        if 'append' in configD['model']:
            append = configD['model']['append']
            args.beam_size = 1

        evaluate_cmd = 'num_process=%s beam_size=%s type=%s bash benchmark/evaluate.sh %s %s %s'%(args.num_process, args.beam_size, args.type, args.s, args.test_fp, args.perform)
        os.system(evaluate_cmd)


main()
