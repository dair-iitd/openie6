from typing import Optional

from overrides import overrides
import torch
import ipdb
from sklearn import metrics

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

import sys
import os

from benchmark import process_outputs, carb

import numpy as np
import json

@Metric.register("carb")
class Carb(Metric):
    """
    Computes scores according to carb framework
    """
    def __init__(self, dev_set: str = None):
        super(Carb, self).__init__()
        self._all_predictions = []
        self._all_confidences = []
        self._all_example_ids = []
        self._append = False
        self._coverage = False
        self._dev_set = dev_set
        self._epoch_num = 0
        # self._train_fp = train_fp
        

    def __call__(self,
                 predictions: list,
                 confidences: list,
                 example_ids: list,
                 append: bool,
                 coverage: bool):
        self._all_predictions.extend(predictions)
        self._all_confidences.extend(confidences)
        self._all_example_ids.extend(example_ids)
        self._append = append
        self._coverage = coverage

    def get_metric(self, reset: bool = False):
        if reset:
            self._epoch_num += 1
            if self._dev_set == 'dev':
                dev_sents_file = os.path.abspath('data/dev/carb_sentences.txt')
            elif self._dev_set == 'test':
                dev_sents_file = os.path.abspah('data/test/carb_sentences.txt')
            dev_sents = open(dev_sents_file,'r').readlines()
            input_lines=[]
            for pred, conf in zip(self._all_predictions, self._all_confidences):
                json_acceptable_str = [[token.replace("'", "\'").replace('"', '\\"') for token in p] for p in pred]
                d = {}
                d["predicted_tokens"] = json_acceptable_str
                d["predicted_log_probs"] = conf.tolist()
                input_lines.append(json.dumps(d))

            # reorder dev sents according to the order of prediction
            dev_sents = [dev_sents[example_id] for example_id in self._all_example_ids]

            if self._append:
                output_lines, rerank_lines, append_lines = process_outputs.process_append(input_lines, dev_sents)
            elif self._coverage:
                output_lines = process_outputs.process_coverage(input_lines, dev_sents)
            else:
                output_lines = process_outputs.process_single(input_lines, dev_sents, threshold=None)

            matchingFunc = carb.Matcher.binary_linient_tuple_match
            if self._dev_set == 'dev':
                dev_pred_file = os.path.abspath('data/dev/carb/extractions.tsv')
            elif self._dev_set == 'test':
                dev_pred_file = os.path.abspath('data/test/carb/extractions.tsv')
            
            if output_lines.strip() == '':
                return {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0} 
                
            # evaluate outputs using carb
            predicted = carb.AllennlpReader(threshold = None)
            predicted.read(output_lines)

            b = carb.Benchmark(dev_pred_file)
            out_filename = "/dev/null"

            auc, optimal_f1_point, _ = b.compare(predicted = predicted.oie,
                                    matchingFunc = matchingFunc,
                                    output_fn = out_filename,
                                    error_file = None,
                                    binary = False)
            print("AUC: {}\t Optimal (precision, recall, F1): {}".format( auc, optimal_f1_point ))
            self.reset()
            return {'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_sum': (auc+optimal_f1_point[2])}

        else:
            return {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    @overrides
    def reset(self):
        self._all_predictions = []
        self._all_confidences = []
        self._all_example_ids = []
