import sys
sys.path.insert(0, 'carb')
import os
import ipdb
import random
import numpy as np
import pickle
import copy
from typing import Dict
from collections import OrderedDict
import logging
from tqdm import tqdm
import regex as re

import torch
import torch.nn.functional as F
from torchtext import data
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import LSTM, CrossEntropyLoss
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import AdamW, AutoModel

from oie_readers.extraction import Extraction
import data
import metric
from metric import Conjunction, Carb

import threading
from threading import Thread
sem = threading.Semaphore()

# prevents printing of model weights, etc
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

def set_seed(seed):
    # Be warned that even with all these seeds, complete reproducibility cannot be guaranteed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return

class Model(pl.LightningModule):

    def __init__(self, hparams, meta_data_vocab=None):
        super(Model, self).__init__()
        self.hparams = hparams

        self._base_model = AutoModel.from_pretrained(
            hparams.model_str, cache_dir='data/pretrained_cache')
        self._hidden_size = self._base_model.config.hidden_size

        if hparams.iterative_layers != 0:
            self._iterative_transformer = self._base_model.encoder.layer[-hparams.iterative_layers:]
            self._base_model.encoder.layer = self._base_model.encoder.layer[:-
                                                                            hparams.iterative_layers]
        else:
            self._iterative_transformer = []

        self._num_labels = 6
        self._dropout = nn.Dropout(p=self.hparams.dropout)

        self._label_embeddings = nn.Embedding(100, self._hidden_size)

        self._labelling_dim = self.hparams.labelling_dim
        self._labelling_layer = nn.Linear(self._labelling_dim, self._num_labels)
        self._merge_layer = nn.Linear(self._hidden_size, self._labelling_dim)

        self._loss = nn.CrossEntropyLoss()

        self._metric = Carb(hparams) if hparams.task == 'oie' else Conjunction()
        self._max_depth = 5 if hparams.task == 'oie' else 3

        self._meta_data_vocab = meta_data_vocab
        self._constD = dict()

        self.all_predictions_conj = []
        self.all_sentence_indices_conj = []
        self.all_conjunct_words_conj = []
        self.all_predictions_oie = []

    def configure_optimizers(self):
        all_params = list(self.named_parameters())
        bert_params = []
        other_params = []
        no_decay = ["bias", "gamma", "beta"]
        params = [{"params": [p for n, p in all_params if not any(nd in n for nd in no_decay) and 'base_model' in n],
                   "weight_decay_rate": 0.01, 'lr': self.hparams.lr},
                  {"params": [p for n, p in all_params if any(nd in n for nd in no_decay) and 'base_model' in n],
                   "weight_decay_rate": 0.0, 'lr': self.hparams.lr},
                  {"params": [p for n, p in all_params if 'base_model' not in n], 'lr': self.hparams.lr}]
        if self.hparams.optimizer == 'adamW':
            optimizer = AdamW(params, lr=1e-3)
        elif self.hparams.optimizer == 'adam':
            optimizer = Adam(params, lr=1e-3)

        if self.hparams.multi_opt and self.hparams.constraints != None:
            num_optimizers = len(self.hparams.constraints.split('_'))
            return [optimizer]*num_optimizers
        else:
            return [optimizer]

    def forward(self, batch, mode='train', batch_idx=-1, constraints=None, cweights=None):
        if self.hparams.wreg != 0 and not hasattr(self, '_initial_parameters'):
            self._initial_parameters = copy.deepcopy(
                dict(self.named_parameters()))

        batch_size, depth, labels_length = batch.labels.shape
        if mode != 'train':
            depth = self._max_depth

        loss, lstm_loss = 0, 0
        hidden_states, _ = self._base_model(batch.text)
        output_dict = dict()
        # (batch_size, seq_length, max_depth, num_labels)
        all_depth_scores = []

        d = 0
        while True:
            for layer in self._iterative_transformer:
                hidden_states = layer(hidden_states)[0]

            hidden_states = self._dropout(hidden_states)
            word_hidden_states = torch.gather(hidden_states, 1, batch.word_starts.unsqueeze(2).repeat(1, 1, hidden_states.shape[2]))

            if d != 0:
                greedy_labels = torch.argmax(word_scores, dim=-1)      
                label_embeddings = self._label_embeddings(greedy_labels)
                word_hidden_states = word_hidden_states + label_embeddings

            word_hidden_states = self._merge_layer(word_hidden_states)
            word_scores = self._labelling_layer(word_hidden_states)
            all_depth_scores.append(word_scores)

            d += 1
            if d >= depth:
                break
            if self.hparams.mode != 'train':
                predictions = torch.max(word_scores, dim=2)[1]
                valid_ext = False
                for p in predictions:
                    if 1 in p and 2 in p:
                        valid_ext = True
                        break
                if not valid_ext:
                    break 
        
        # (batch_size, seq_length, max_depth)
        all_depth_predictions, all_depth_confidences = [], []
        batch_size, num_words, _ = word_scores.shape
        batch.labels = batch.labels.long()
        for d, word_scores in enumerate(all_depth_scores):
            if mode == 'train':
                batch_labels_d = batch.labels[:, d, :]
                mask = torch.ones(batch.word_starts.shape).int().type_as(hidden_states)
                loss += self._loss(word_scores.reshape(batch_size*num_words, -1), batch.labels[:, d, :].reshape(-1))
            else:
                word_log_probs = torch.log_softmax(word_scores, dim=2)
                max_log_probs, predictions = torch.max(word_log_probs, dim=2)

                padding_labels = (batch.labels[:,0,:]!=-100).float()

                sro_label_predictions = (predictions!=0).float() * padding_labels
                log_probs_norm_ext_len = (max_log_probs * sro_label_predictions) / (sro_label_predictions.sum(dim=0)+1)
                confidences = torch.exp(torch.sum(log_probs_norm_ext_len, dim=1))

                all_depth_predictions.append(predictions.unsqueeze(1))
                all_depth_confidences.append(confidences.unsqueeze(1))

        if mode == 'train':
            if constraints != '':
                all_depth_scores = torch.cat([d.unsqueeze(1) for d in all_depth_scores], dim=1)
                all_depth_scores = torch.softmax(all_depth_scores, dim=-1)

                const_loss = self.constrained_loss(all_depth_scores, batch, constraints, cweights) / batch_size
                loss = const_loss

            if self.hparams.wreg != 0:
                weight_diff = 0
                current_parameters = dict(self.named_parameters())
                for name in self._initial_parameters:
                    weight_diff += torch.norm(
                        current_parameters[name]-self._initial_parameters[name])
                loss = loss + self.hparams.wreg*weight_diff
        else:
            
            all_depth_predictions = torch.cat(all_depth_predictions, dim=1)
            all_depth_confidences = torch.cat(all_depth_confidences, dim=1)
                
            output_dict['predictions'] = all_depth_predictions
            output_dict['scores'] = all_depth_confidences

            if constraints != '' and 'predict' not in self.hparams.mode and self.hparams.batch_size != 1:
                all_depth_scores = torch.cat([d.unsqueeze(1) for d in all_depth_scores], dim=1)
                all_depth_scores.fill_(0)

                # labels = copy.copy(batch.labels) # for checking test set
                # labels[labels == -100] = 0
                labels = copy.copy(all_depth_predictions)

                labels = labels.unsqueeze(-1)
                labels_depth = labels.shape[1]
                all_depth_scores = all_depth_scores[:, :labels_depth, :, :]
                all_depth_scores.scatter_(3, labels.long(), 1)

                constraints, cweights = 'posm_hvc_hvr_hve', '1_1_1_1'
                constraints_list, cweights_list = constraints.split(
                    '_'), cweights.split('_')
                if len(constraints_list) != len(cweights_list):
                    cweights_list = [cweights]*len(constraints_list)

                for constraint, weight in zip(constraints_list, cweights_list):
                    const_loss = self.constrained_loss(all_depth_scores, batch, constraint, float(weight))
                    if constraint not in self._constD:
                        self._constD[constraint] = []
                    self._constD[constraint].append(const_loss)

        output_dict['loss'] = loss
        return output_dict

    def constrained_loss(self, all_depth_scores, batch, constraints, cweights):
        batch_size, depth, num_words, labels = all_depth_scores.shape
        hinge_loss = 0
        verb_scores = torch.gather(all_depth_scores, 2, batch.verb_index.unsqueeze(1).unsqueeze(3).repeat(1, depth, 1, labels))
        verb_rel_scores = verb_scores[:, :, :, 2]
        # (batch_size, depth, num_words)
        verb_rel_scores = verb_rel_scores * (batch.verb_index != 0).unsqueeze(1).float()

        # every head-verb must be included in a relation
        if 'hvc' in constraints:
            column_loss = torch.abs(1-torch.sum(verb_rel_scores, dim=1))
            column_loss = column_loss[batch.verb_index != 0]
            hinge_loss += cweights*column_loss.sum()

        # extractions must have atleast k-relations with a head verb in them
        if 'hvr' in constraints:
            row_rel_loss = F.relu(batch.verb.sum(dim=1).float() - torch.max(verb_rel_scores, dim=2)[0].sum(dim=1))
            hinge_loss += cweights*row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in constraints:
            ex_loss = F.relu(torch.sum(verb_rel_scores, dim=2)-1)
            hinge_loss += cweights*ex_loss.sum()

        if 'posm' in constraints:
            pos_scores = torch.gather(all_depth_scores, 2, batch.pos_index.unsqueeze(1).unsqueeze(3).repeat(1, depth, 1, labels))
            pos_nnone_scores = torch.max(pos_scores[:, :, :, 1:], dim=-1)[0]
            column_loss = (1-torch.max(pos_nnone_scores, dim=1)[0]) * (batch.pos_index != 0).float()
            hinge_loss += cweights*column_loss.sum()

        return hinge_loss

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        if type(self.trainer.checkpoint_callback.kth_value) != type(0.0):
            best = self.trainer.checkpoint_callback.kth_value.item()
        else:
            best = self.trainer.checkpoint_callback.kth_value
        tqdm = {'loss': '{:.3f}'.format(avg_training_loss), 'best': best}
        return tqdm

    def training_step(self, batch, batch_idx, optimizer_idx=-1):
        batch = data.dotdict(batch)

        if self.hparams.multi_opt:
            constraints = self.hparams.constraints.split('_')[optimizer_idx]
            cweights = float(self.hparams.cweights.split('_')[optimizer_idx])
        else:
            constraints = self.hparams.constraints
            cweights = float(self.hparams.cweights)

        output_dict = self.forward(batch, mode='train', batch_idx=batch_idx, constraints=constraints, cweights=cweights)

        tqdm_dict = {"train_loss": output_dict['loss']}
        output = OrderedDict({"loss": output_dict['loss'], "log": tqdm_dict})

        return output

    def validation_step(self, batch, batch_idx):
        batch = data.dotdict(batch)
        output_dict = self.forward(batch, mode='val', constraints=self.hparams.constraints, cweights=self.hparams.cweights)

        outputD = {"predictions": output_dict['predictions'], "scores": output_dict['scores'],
                   "ground_truth": batch.labels, "meta_data": batch.meta_data}
        output = OrderedDict(outputD)

        if self.hparams.mode!='test':
            if self.hparams.write_async:
                t = Thread(target=self.write_to_file, args=(output, batch_idx, self.hparams.task))
                t.start() 
            else:
                self.write_to_file(output, batch_idx, self.hparams.task)

        return output

    def evaluation_end(self, outputs, mode):
        result = None
        if self.hparams.mode == 'test':
            for output_index, output in enumerate(outputs):
                output['predictions'] = output['predictions'].cpu()
                output['scores'] = output['scores'].cpu()
                output['scores'] = (output['scores'] * 100).round() / 100
                output['ground_truth'] = output['ground_truth'].cpu()
                output['meta_data'] = output['meta_data'].cpu()
        if self.hparams.task == 'conj':
            if 'predict' in self.hparams.mode:
                metrics = {'P_exact': 0, 'R_exact': 0, 'F1_exact': 0}
            else:
                for output in outputs:
                    if type(output['meta_data'][0]) != type(""):
                        output['meta_data'] = [self._meta_data_vocab.itos[m] for m in output['meta_data']]
                    self._metric(output['predictions'], output['ground_truth'], meta_data=output['meta_data'])
                metrics = self._metric.get_metric(reset=True, mode=mode)

            val_acc, val_auc = metrics['F1_exact'], 0
            result = {"eval_f1": val_acc, "eval_p": metrics['P_exact'], "eval_r": metrics['R_exact']}

        elif self.hparams.task == 'oie':
            if 'predict' in self.hparams.mode:
                metrics = {'carb_f1': 0, 'carb_auc': 0, 'carb_lastf1': 0}
            else:
                for output in outputs:
                    if type(output['meta_data'][0]) != type(""):
                        output['meta_data'] = [self._meta_data_vocab.itos[m] for m in output['meta_data']]
                    self._metric(output['predictions'], output['meta_data'], output['scores'])
                metrics = self._metric.get_metric(reset=True, mode=mode)

            result = {"eval_f1": metrics['carb_f1'], "eval_auc": metrics['carb_auc'], "eval_lastf1": metrics['carb_lastf1']}
            
        print('\nResults: '+str(result))
        # For computing the constraint violations
        # if hasattr(self, '_constD') and self.hparams.constraints != '':
        #     for key in self._constD:
        #         self._constD[key] = sum(self._constD[key]).item()
        #     print('\nViolations: ', self._constD)
        #     self._constD = dict()
        return result

    def validation_epoch_end(self, outputs):
        eval_results = self.evaluation_end(outputs, 'dev')
        result = {}
        if eval_results != None:
            result = {"log": eval_results, "eval_acc": eval_results['eval_f1']}

        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        eval_results = self.evaluation_end(outputs, 'test')
        self.outputs = outputs
        result = {"log": eval_results, "progress_bar": eval_results,
                  "test_acc": eval_results['eval_f1']}
        self.results = eval_results
        if self.hparams.write_async:
            while not sem.acquire(blocking=True):
                pass
            sem.release()
        return result

    # obligatory definitions - pass actual through fit
    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def write_to_file(self, output, batch_idx,task):
        if self.hparams.write_async:
            while not sem.acquire(blocking=True):
                # print("No Semaphore available")
                pass
            # print('Got semaphore')
        output['predictions'] = output['predictions'].cpu()
        output['scores'] = output['scores'].cpu()
        output['ground_truth'] = output['ground_truth'].cpu()
        output['meta_data'] = output['meta_data'].cpu()
        def process_extraction(extraction, sentence, score):
            # rel, arg1, arg2, loc, time = [], [], [], [], []
            rel, arg1, arg2, loc_time, args = [], [], [], [], []
            tag_mode = 'none'
            rel_case = 0
            for i, token in enumerate(sentence):
                if '[unused' in token:
                    if extraction[i].item() == 2:
                        rel_case = int(re.search('\[unused(.*)\]', token).group(1))
                    continue
                if extraction[i] == 1:
                    arg1.append(token)
                if extraction[i] == 2:
                    rel.append(token)
                if extraction[i] == 3:
                    arg2.append(token)
                if extraction[i] == 4:
                    loc_time.append(token)

            rel = ' '.join(rel).strip()
            if rel_case == 1:
                rel = 'is '+rel
            elif rel_case == 2:
                rel = 'is '+rel+' of'
            elif rel_case == 3:
                rel = 'is '+rel+' from'

            arg1 = ' '.join(arg1).strip()
            arg2 = ' '.join(arg2).strip()
            args = ' '.join(args).strip()
            loc_time = ' '.join(loc_time).strip()
            if not self.hparams.no_lt:
                arg2 = (arg2+' '+loc_time+' '+args).strip()
            sentence_str = ' '.join(sentence).strip()

            extraction = Extraction(pred=rel, head_pred_index=None, sent=sentence_str, confidence=score, index=0)
            extraction.addArg(arg1)
            extraction.addArg(arg2)

            return extraction
        def contains_extraction(extr, list_extr):
            str = ' '.join(extr.args) + ' ' + extr.pred
            for extraction in list_extr:
                if str == ' '.join(extraction.args) + ' ' + extraction.pred:
                    return True
            return False
        output['meta_data'] = [self._meta_data_vocab.itos[m] for m in output['meta_data']]
        if task == 'oie':
            predictions = output['predictions']
            sentences = output['meta_data']
            scores = output['scores']
            num_sentences, extractions, max_sentence_len = predictions.shape
            assert num_sentences == len(sentences)
            all_predictions = {}
            for i, sentence_str in enumerate(sentences):
                words = sentence_str.split() + \
                    ['[unused1]', '[unused2]', '[unused3]']
                orig_sentence = sentence_str.split('[unused1]')[0].strip()
                if self._metric.mapping:
                    if self._metric.mapping[orig_sentence] not in all_predictions:
                        all_predictions[self._metric.mapping[orig_sentence]] = []
                else:
                    if orig_sentence not in all_predictions:
                        all_predictions[orig_sentence] = []
                for j in range(extractions):
                    extraction = predictions[i][j][:len(words)]
                    if sum(extraction) == 0:  # extractions completed
                        break
                    pro_extraction = process_extraction(
                        extraction, words, scores[i][j].item())
                    if pro_extraction.args[0] != '' and pro_extraction.pred != '':
                        if self._metric.mapping:
                            if not contains_extraction(pro_extraction, all_predictions[self._metric.mapping[orig_sentence]]):
                                all_predictions[self._metric.mapping[orig_sentence]].append(
                                    pro_extraction)
                        else:
                            if not contains_extraction(pro_extraction, all_predictions[orig_sentence]):
                                all_predictions[orig_sentence].append(pro_extraction)
            all_pred = []
            all_pred_allennlp = []
            for example_id, sentence in enumerate(all_predictions):
                predicted_extractions = all_predictions[sentence]
                # if 'predict' in self.hparams.mode: # write only the results in text file
                sentence_str = f'{sentence}\n'
                for extraction in predicted_extractions:
                    if self.hparams.type == 'sentences':
                        ext_str = data.ext_to_sentence(extraction) + '\n'
                    else:
                        ext_str = data.ext_to_string(extraction) + '\n'
                    sentence_str += ext_str
                all_pred.append(sentence_str)
                sentence_str_allennlp = ''
                for extraction in predicted_extractions:
                    args1 = ' '.join(extraction.args[1:])
                    ext_str = f'{sentence}\t<arg1> {extraction.args[0]} </arg1> <rel> {extraction.pred} </rel> <arg2> {args1} </arg2>\t{extraction.confidence}\n'
                    sentence_str_allennlp += ext_str
                    sentence_str_allennlp.strip('\n')
                all_pred_allennlp.append(sentence_str_allennlp)
            self.all_predictions_oie.extend(all_pred)
        if task =='conj':
            example_id, correct = 0, True
            total1, total2 = 0, 0
            predictions = output['predictions']
            gt = output['ground_truth']
            meta_data = output['meta_data']
            total_depth = predictions.shape[1]
            all_pred = []
            all_conjunct_words = []
            all_sentence_indices = []
            for idx in range(len(meta_data)):
                example_id += 1
                sentence = meta_data[idx]
                words = sentence.split()
                sentence_predictions, sentence_gt = [], []
                for depth in range(total_depth):
                    depth_predictions = predictions[idx][depth][:len(
                        words)].tolist()
                    sentence_predictions.append(depth_predictions)
                pred_coords = metric.get_coords(sentence_predictions)

                words = sentence.split()
                sentence_str = sentence+'\n'
                split_sentences, conj_words, sentences_indices = data.coords_to_sentences(
                    pred_coords, words)
                all_sentence_indices.append(sentences_indices)
                all_conjunct_words.append(conj_words)
                total1 += len(split_sentences)
                total2 += 1 if len(split_sentences) > 0 else 0
                sentence_str += '\n'.join(split_sentences)+'\n'

                all_pred.append(sentence_str)
            self.all_conjunct_words_conj.extend(all_conjunct_words)
            self.all_predictions_conj.extend(all_pred)
            self.all_sentence_indices_conj.extend(all_sentence_indices)
        if self.hparams.out != None:
            directory = os.path.dirname(self.hparams.out)
            if directory != '' and not os.path.exists(directory):
                os.makedirs(directory)
            out_fp = f'{self.hparams.out}.{self.hparams.task}'
            # print('Predictions written to ', out_fp)
            if batch_idx == 0:
                predictions_f = open(out_fp,'w')
            else:
                predictions_f = open(out_fp,'a')
            predictions_f.write('\n'.join(all_pred)+'\n')
            predictions_f.close()
        if task == 'oie' and self.hparams.write_allennlp:
            if batch_idx == 0:
                predictions_f_allennlp = open(f'{self.hparams.out}.allennlp', 'w')
                self.predictions_f_allennlp = predictions_f_allennlp.name
            else:
                predictions_f_allennlp = open(f'{self.hparams.out}.allennlp', 'a')
            predictions_f_allennlp.write(''.join(all_pred_allennlp))
            predictions_f_allennlp.close()
        if self.hparams.write_async:
            sem.release()
