from carb import Benchmark
from oie_readers.extraction import Extraction
from matcher import Matcher

from collections import defaultdict
from enum import Enum
from tqdm import tqdm

import os
import ipdb
import pickle
from overrides import overrides
import numpy as np
import warnings
import regex as re
import difflib


class Record(object):
    """
    the precision equals how many of the conjuncts output
    by the algorithm are correct, and the recall is the
    percentage of conjuncts found by the algorithm.
    [Shimbo et al, 2007]
    """

    def __init__(self):
        self.tp_t = 0
        self.tp_f = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @property
    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp_t + self.tn) / total if total > 0 else np.nan

    @property
    def precision(self):
        denom = self.tp + self.fp
        return self.tp_t / denom if denom > 0 else np.nan

    @property
    def recall(self):
        denom = self.tp + self.fn
        return self.tp_t / denom if denom > 0 else np.nan

    @property
    def f1_score(self):
        precision = self.precision
        if precision is not np.nan:
            recall = self.recall
            if recall is not np.nan:
                denom = precision + recall
                if denom > 0:
                    return (2 * precision * recall) / denom
        return np.nan

    def __str__(self):
        return "P: {:.8f}, R: {:.8f}, F: {:.8f}" \
            .format(self.precision, self.recall, self.f1_score)

    def __repr__(self):
        return "Record(TP=({},t:{},f:{}), FP={}, FN={}, TN={})" \
            .format(self.tp, self.tp_t, self.tp_f, self.fp, self.fn, self.tn)


class Coordination(object):
    __slots__ = ('cc', 'conjuncts', 'seps', 'label')

    def __init__(self, cc, conjuncts, seps=None, label=None):
        assert isinstance(conjuncts, (list, tuple)) and len(conjuncts) >= 2
        assert all(isinstance(conj, tuple) for conj in conjuncts)
        conjuncts = sorted(conjuncts, key=lambda span: span[0])
        # NOTE(chantera): The form 'A and B, C' is considered to be coordination.  # NOQA
        # assert cc > conjuncts[-2][1] and cc < conjuncts[-1][0]
        assert cc > conjuncts[0][1] and cc < conjuncts[-1][0]
        if seps is not None:
            # if len(seps) == len(conjuncts) - 2:
            #     for i, sep in enumerate(seps):
            #         assert conjuncts[i][1] < sep and conjuncts[i + 1][0] > sep
            # else:
            if len(seps) != len(conjuncts) - 2:
                warnings.warn(
                    "Coordination does not contain enough separators. "
                    "It may be a wrong coordination: "
                    "cc={}, conjuncts={}, separators={}"
                    .format(cc, conjuncts, seps))
        else:
            seps = []
        self.cc = cc
        self.conjuncts = tuple(conjuncts)
        self.seps = tuple(seps)
        self.label = label

    def get_pair(self, index, check=False):
        pair = None
        for i in range(1, len(self.conjuncts)):
            if self.conjuncts[i][0] > index:
                pair = (self.conjuncts[i - 1], self.conjuncts[i])
                assert pair[0][1] < index and pair[1][0] > index
                break
        if check and pair is None:
            raise LookupError(
                "Could not find any pair for index={}".format(index))
        return pair

    def __repr__(self):
        return "Coordination(cc={}, conjuncts={}, seps={}, label={})".format(
            self.cc, self.conjuncts, self.seps, self.label)

    def __eq__(self, other):
        if not isinstance(other, Coordination):
            return False
        return self.cc == other.cc \
            and len(self.conjuncts) == len(other.conjuncts) \
            and all(conj1 == conj2 for conj1, conj2
                    in zip(self.conjuncts, other.conjuncts))


def post_process(coords, is_quote):
    new_coords = {}
    offsets = np.delete(is_quote.cumsum(), np.argwhere(is_quote))
    for cc, coord in coords.items():
        cc = cc + offsets[cc]
        if coord is not None:
            conjuncts = [(b + offsets[b], e + offsets[e])
                         for (b, e) in coord.conjuncts]
            seps = [s + offsets[s] for s in coord.seps]
            coord = Coordination(cc, conjuncts, seps, coord.label)
        new_coords[cc] = coord
    return new_coords


class Counter(object):

    class Criteria(Enum):
        WHOLE = 0
        OUTER = 1
        INNER = 2
        EXACT = 3

    OVERALL = "OVERALL"

    def __init__(self, criteria):
        assert isinstance(criteria, Counter.Criteria)
        self._criteria = criteria
        self._records = defaultdict(Record)

    def reset(self):
        self._records.clear()

    def append(self, pred_coords, true_coords):
        for cc in sorted(true_coords.keys()):
            pred_coord = pred_coords.get(cc, None)
            true_coord = true_coords[cc]
            if pred_coord is not None and true_coord is not None:
                pred_conjuncts = pred_coord.conjuncts
                true_conjuncts = true_coord.conjuncts
                coord_label = true_coord.label
                if self._criteria == Counter.Criteria.WHOLE:
                    correct = pred_conjuncts[0][0] == true_conjuncts[0][0] \
                        and pred_conjuncts[-1][1] == true_conjuncts[-1][1]
                elif self._criteria == Counter.Criteria.OUTER:
                    correct = pred_conjuncts[0] == true_conjuncts[0] \
                        and pred_conjuncts[-1] == true_conjuncts[-1]
                elif self._criteria == Counter.Criteria.INNER:
                    pred_pair = pred_coord.get_pair(cc, check=True)
                    true_pair = true_coord.get_pair(cc, check=True)
                    correct = pred_pair == true_pair
                elif self._criteria == Counter.Criteria.EXACT:
                    correct = pred_conjuncts == true_conjuncts
                self._records[Counter.OVERALL].tp += 1
                self._records[coord_label].tp += 1
                if correct:
                    self._records[Counter.OVERALL].tp_t += 1
                    self._records[coord_label].tp_t += 1
                else:
                    self._records[Counter.OVERALL].tp_f += 1
                    self._records[coord_label].tp_f += 1
            if pred_coord is not None and true_coord is None:
                self._records[Counter.OVERALL].fp += 1
            if pred_coord is None and true_coord is not None:
                coord_label = true_coord.label
                self._records[Counter.OVERALL].fn += 1
                self._records[coord_label].fn += 1
            if pred_coord is None and true_coord is None:
                self._records[Counter.OVERALL].tn += 1

    @property
    def overall(self):
        return self._records[Counter.OVERALL]


def clean_conjuncts(coordination, words):
    cc_index = coordination.cc
    conjuncts = coordination.conjuncts
    seps = coordination.seps
    return coordination


def get_coords(all_depth_labels, tokens=None, correct=False):
    all_cps = dict()

    found = False
    for depth in range(len(all_depth_labels)):
        cp, start_index = None, -1
        coordphrase, conjunction, coordinator, separator = False, False, False, False
        labels = all_depth_labels[depth]

        for i, label in enumerate(labels):
            if label != 1:  # conjunction can end
                if conjunction and cp != None:
                    conjunction = False
                    cp['conjuncts'].append((start_index, i-1))
            if label == 0 or label == 2:  # coordination phrase can end
                if cp != None and len(cp['conjuncts']) >= 2 and cp['cc'] > cp['conjuncts'][0][1] and cp['cc'] < cp['conjuncts'][-1][0]:
                    found = True
                    coordination = Coordination(
                        cp['cc'], cp['conjuncts'], label=depth)
                    # if correct:
                    #     coordination = clean_conjuncts(coordination, words)
                    all_cps[cp['cc']] = coordination
                    cp = None

            if label == 0:
                continue
            if label == 1:  # can start a conjunction
                if not conjunction:
                    conjunction = True
                    start_index = i
            if label == 2:  # starts a coordination phrase
                cp = {'cc': -1, 'conjuncts': [], 'seps': []}
                conjunction = True
                start_index = i
            if label == 3 and cp != None:
                cp['cc'] = i
            if label == 4 and cp != None:
                cp['seps'].append(i)
            if label == 5:  # nothing to be done
                continue
            if label == 3 and cp == None:
                # coordinating words which do not have associated conjuncts
                all_cps[i] = None

    return all_cps


def contains_extraction(extr, list_extr):
    str = ' '.join(extr.args) + ' ' + extr.pred
    for extraction in list_extr:
        if str == ' '.join(extraction.args) + ' ' + extraction.pred:
            return True
    return False


def dedup_extractions(extractions_list, conj_words):
    # Remove extractions which are exactly equal
    # Remove extractions which are almost equal (threshold=0.9) - ignoring extractions which are actually split (as they may be very similar)
    all_ext_words = []
    for extr in extractions_list:
        ext_words = (' '.join(extr.args) + ' ' + extr.pred).split()
        all_ext_words.append(ext_words)

    delete_indices = []
    conj_words_set = set(conj_words)
    for i in range(len(all_ext_words)):
        for j in range(i+1, len(all_ext_words)):
            ext_i_str = ' '.join(all_ext_words[i])
            ext_j_str = ' '.join(all_ext_words[j])
            if ext_i_str == ext_j_str:
                delete_indices.append(i)
                continue
            ext_i_set = set(all_ext_words[i])
            ext_j_set = set(all_ext_words[j])
            len_i = len(ext_i_set)
            len_j = len(ext_j_set)
            found_conjunction = False
            for conj_words in conj_words_set:
                if conj_words in ext_i_str or conj_words in ext_j_str:
                    found_conjunction = True
            if found_conjunction:
                continue
            if difflib.SequenceMatcher(None, ext_i_str, ext_j_str).ratio() > 0.9:
                if len_i > len_j:
                    delete_indices.append(j)
                else:
                    delete_indices.append(i)

    delete_indices = list(set(delete_indices))
    for index in sorted(delete_indices, reverse=True):
        del extractions_list[index]
    return extractions_list


class Conjunction():
    def __init__(self, dump_dir=None):
        super(Conjunction, self).__init__()
        self._counter_whole = Counter(Counter.Criteria.WHOLE)
        self._counter_outer = Counter(Counter.Criteria.OUTER)
        self._counter_inner = Counter(Counter.Criteria.INNER)
        self._counter_exact = Counter(Counter.Criteria.EXACT)
        self.n_complete = 0
        self.n_sentence = 0
        self._dump_dir = dump_dir
        if self._dump_dir != None:
            if os.path.exists(dump_dir+'/tokens.pkl'):
                os.remove(dump_dir+'/tokens.pkl')
            if os.path.exists(dump_dir+'/pred_it_coords.pkl'):
                os.remove(dump_dir+'/pred_it_coords.pkl')
            if os.path.exists(dump_dir+'/gt_it_coords.pkl'):
                os.remove(dump_dir+'/gt_it_coords.pkl')

    def __call__(self, predictions, ground_truth, meta_data=None, coords=False):
        # coords == True when we give it the complete coords
        # happens when we want to evaluate on the original system outputs
        for i in range(len(predictions)):
            if not coords:
                pred_coords = get_coords(
                    predictions[i], meta_data[i], correct=True)
                true_coords = get_coords(ground_truth[i], meta_data[i])
            else:
                pred_coords = predictions[i]
                true_coords = ground_truth[i]

            self._counter_whole.append(pred_coords, true_coords)
            self._counter_outer.append(pred_coords, true_coords)
            self._counter_inner.append(pred_coords, true_coords)
            self._counter_exact.append(pred_coords, true_coords)

            if self._dump_dir:
                pickle.dump(tokens, open(self._dump_dir+'/tokens.pkl', 'ab'))
                pickle.dump(pred_coords, open(
                    self._dump_dir+'/pred_it_coords.pkl', 'ab'))
                pickle.dump(true_coords, open(
                    self._dump_dir+'/gt_it_coords.pkl', 'ab'))
        return

    def reset(self):
        self._counter_whole.reset()
        self._counter_outer.reset()
        self._counter_inner.reset()
        self._counter_exact.reset()
        self.n_complete = 0
        self.n_sentence = 0

    def get_metric(self, reset: bool = False, mode=None):
        counters = [("whole", self._counter_whole),
                    ("outer", self._counter_outer),
                    ("inner", self._counter_inner),
                    ("exact", self._counter_exact)]

        all_metrics = dict()
        all_metrics['P_exact'] = counters[3][1].overall.precision
        all_metrics['R_exact'] = counters[3][1].overall.recall
        all_metrics['F1_whole'] = counters[0][1].overall.f1_score
        all_metrics['F1_outer'] = counters[1][1].overall.f1_score
        all_metrics['F1_inner'] = counters[1][1].overall.f1_score
        all_metrics['F1_exact'] = counters[3][1].overall.f1_score
        if reset:
            self.reset()
        return all_metrics

    def get_overall_score(self, metric='exact'):
        if metric == 'whole':
            counter = self._counter_whole
        elif metric == 'outer':
            counter = self._counter_outer
        elif metric == 'inner':
            counter = self._counter_inner
        elif metric == 'exact':
            counter = self._counter_exact
        else:
            raise ValueError('invalid metric: {}'.format(metric))
        return counter.overall.f1_score


class Carb():
    def __init__(self, hparams, mapping=None):
        super(Carb, self).__init__()
        self._dev_benchmark = Benchmark('carb/data/gold/dev.tsv')
        self._test_benchmark = Benchmark('carb/data/gold/test.tsv')
        self.matchingFunc = Matcher.binary_linient_tuple_match
        self._all_predictions, self._all_pos_words, self._all_verb_words = {}, {}, {}
        self.score = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}
        self.hparams = hparams
        self.num_extractions = self.hparams.num_extractions
        self.mapping = None
        self.conj_word_mapping = None

    def __call__(self, predictions, sentences, scores, pos_words=None, verb_words=None):
        num_sentences, extractions, max_sentence_len = predictions.shape
        assert num_sentences == len(sentences)

        for i, sentence_str in enumerate(sentences):
            words = sentence_str.split() + ['[unused1]', '[unused2]', '[unused3]']
            orig_sentence = sentence_str.split('[unused1]')[0].strip()
            if self.mapping:
                if self.mapping[orig_sentence] not in self._all_predictions:
                    self._all_predictions[self.mapping[orig_sentence]] = []
            else:
                if orig_sentence not in self._all_predictions:
                    self._all_predictions[orig_sentence] = []
            if pos_words != None:
                self._all_pos_words[orig_sentence] = pos_words[i]
            if verb_words != None:
                self._all_verb_words[orig_sentence] = verb_words[i]

            for j in range(extractions):
                extraction = predictions[i][j][:len(words)]
                if sum(extraction) == 0:  # extractions completed
                    break
                pro_extraction = self._process_extraction(
                    extraction, words, scores[i][j].item())
                if pro_extraction.args[0] != '' and pro_extraction.pred != '':
                    if self.mapping:
                        if not contains_extraction(pro_extraction, self._all_predictions[self.mapping[orig_sentence]]):
                            self._all_predictions[self.mapping[orig_sentence]].append(
                                pro_extraction)
                    else:
                        if not contains_extraction(pro_extraction, self._all_predictions[orig_sentence]):
                            self._all_predictions[orig_sentence].append(pro_extraction)

        # if self.mapping or self.conj_word_mapping:
        #     for sentence in self._all_predictions:
        #         dextractions = dedup_extractions(
        #             self._all_predictions[sentence], self.conj_word_mapping[sentence])
        #         self._all_predictions[sentence] = dextractions

        return

    def get_metric(self, reset, mode):
        if self.num_extractions:
            for sentence in self._all_predictions:
                self._all_predictions[sentence] = sorted(self._all_predictions[sentence],
                                                         key=lambda x: x.confidence, reverse=True)[:self.num_extractions]

        out_filename = "/dev/null"
        if mode == 'dev':
            auc, optimal_f1_point, last_f1_point = self._dev_benchmark.compare(predicted=self._all_predictions,
                                                                               matchingFunc=self.matchingFunc,
                                                                               output_fn=out_filename, error_file=None,
                                                                               binary=False)
        elif mode == 'test':
            auc, optimal_f1_point, last_f1_point = self._test_benchmark.compare(predicted=self._all_predictions,
                                                                                matchingFunc=self.matchingFunc,
                                                                                output_fn=out_filename, error_file=None,
                                                                                binary=False)
        else:
            assert False

        self.score = {
            'carb_auc': auc, 'carb_f1': optimal_f1_point[2], 'carb_lastf1': last_f1_point[2]}
        score = self.score
        if mode == 'dev' and reset:
            self.reset()
        return score

    def reset(self):
        self._all_predictions = {}
        self.score = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    def _process_extraction(self, extraction, sentence, score):
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

        extraction = Extraction(
            pred=rel, head_pred_index=None, sent=sentence_str, confidence=score, index=0)
        extraction.addArg(arg1)
        extraction.addArg(arg2)

        return extraction

    def _process_allenlp_format(self, lines):
        assert self._all_predictions == {}
        for line in lines:
            extr = line.split('\t')
            sentence = extr[0]
            confidence = float(extr[2])
            
            arg1 = re.findall("<arg1>.*</arg1>", extr[1])[0].strip('<arg1>').strip('</arg1>').strip()
            rel = re.findall("<rel>.*</rel>", extr[1])[0].strip('<rel>').strip('</rel>').strip()
            arg2 = re.findall("<arg2>.*</arg2>", extr[1])[0].strip('<arg2>').strip('</arg2>').strip()
            
            extraction = Extraction(pred=rel, head_pred_index=None, sent=sentence, confidence=confidence, index=0)
            extraction.addArg(arg1)
            extraction.addArg(arg2)

            if sentence not in self._all_predictions:
                self._all_predictions[sentence] = []
            self._all_predictions[sentence] = extraction
