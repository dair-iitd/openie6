'''
Usage:
   benchmark --gold=GOLD_OIE --out=OUTPUT_FILE (--openiefive=OPENIE5 | --allennlp=ALLENNLP | --stanford=STANFORD_OIE | --ollie=OLLIE_OIE |--reverb=REVERB_OIE | --clausie=CLAUSIE_OIE | --openiefour=OPENIEFOUR_OIE | --props=PROPS_OIE | --tabbed=TABBED_OIE) [--exactMatch | --predMatch | --argMatch] [--error-file=ERROR_FILE] [--threshold=THRESHOLD] [--pickle_output=PICKLE_OUTPUT]

Options:
  --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
  --out-OUTPUT_FILE            The output file, into which the precision recall curve will be written.
  --clausie=CLAUSIE_OIE        Read ClausIE format from file CLAUSIE_OIE.
  --ollie=OLLIE_OIE            Read OLLIE format from file OLLIE_OIE.
  --openiefour=OPENIEFOUR_OIE  Read Open IE 4 format from file OPENIEFOUR_OIE.
  --openiefive=OPENIE5         Read Open IE 5 format from file OPENIE5.
  --props=PROPS_OIE            Read PropS format from file PROPS_OIE
  --reverb=REVERB_OIE          Read ReVerb format from file REVERB_OIE
  --stanford=STANFORD_OIE      Read Stanford format from file STANFORD_OIE
  --tabbed=TABBED_OIE          Read simple tab format file, where each line consists of:
                                sent, prob, pred,arg1, arg2, ...
  --allennlp=ALLENNLP_OIE      Read from allennlp output format
  --exactmatch                 Use exact match when judging whether an extraction is correct.
'''
import docopt
import string
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import re
import logging
import pdb
import ipdb
import pickle
logging.basicConfig(level = logging.INFO)

from oie_readers.stanfordReader import StanfordReader
from oie_readers.ollieReader import OllieReader
from oie_readers.reVerbReader import ReVerbReader
from oie_readers.clausieReader import ClausieReader
from oie_readers.openieFourReader import OpenieFourReader
from oie_readers.openieFiveReader import OpenieFiveReader
from oie_readers.propsReader import PropSReader
from oie_readers.tabReader import TabReader
from oie_readers.allennlpReader import AllennlpReader
# from oie_readers.rnnoieReader import RnnOIEReader

from oie_readers.goldReader import GoldReader
from matcher import Matcher
from operator import itemgetter

class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''
    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        gr = GoldReader()
        gr.read(gold_fn)
        self.gold = gr.oie

    def compare(self, predicted, matchingFunc, output_fn, error_file = None, pickle_output_fp=None):
        ''' Compare gold against predicted using a specified matching function.
            Outputs PR curve to output_fn '''

        y_true = []
        y_scores = []
        errors = []
        correct = 0
        incorrect = 0

        correctTotal = 0
        unmatchedCount = 0
        
        predicted = Benchmark.normalizeDict(predicted)
        gold = Benchmark.normalizeDict(self.gold)
        
        pickle_output = {}

        for sent, item in gold.items():
            goldExtractions, goldSent = item['extractions'], item['orig_sent']
            sentence_y_true, sentence_y_scores = [], []
            sentence_unmatchedCount = 0
            if sent not in predicted:
                # The extractor didn't find any extractions for this sentence
                # for goldEx in goldExtractions:
                sentence_unmatchedCount += len(goldExtractions)
                unmatchedCount += len(goldExtractions)
                correctTotal += len(goldExtractions)
                continue

            predictedExtractions, predictedSent = predicted[sent]['extractions'], predicted[sent]['orig_sent']
            for goldEx in goldExtractions:
                correctTotal += 1
                found = False

                #print goldEx.bow()
                for predictedEx in predictedExtractions:
                    if output_fn in predictedEx.matched:
                        # This predicted extraction was already matched against a gold extraction
                        # Don't allow to match it again
                        continue

                    if matchingFunc(goldEx,
                                    predictedEx,
                                    ignoreStopwords = True,
                                    ignoreCase = True):
                        # print('***Matching***')
                        # print(goldEx.pred, goldEx.args)
                        # print(predictedEx.pred, predictedEx.args)

                        sentence_y_true.append(1)
                        sentence_y_scores.append(predictedEx.confidence)
                        predictedEx.matched.append(output_fn)
                        found = True
                        correct += 1
                        # print("Correct: ", predictedEx.confidence, predictedEx.bow())
                        # print "Correct: ", '\t'.join([predictedEx.pred] + predictedEx.args) 
                        break

                if not found:
                    # print("Unmatched: ", '\t'.join([goldEx.pred] + goldEx.args))
                    errors.append(goldEx.index)
                    sentence_unmatchedCount += 1

            unmatchedCount += sentence_unmatchedCount

            for predictedEx in [x for x in predictedExtractions if (output_fn not in x.matched)]:
                # Add false positives
                sentence_y_true.append(0)
                incorrect+=1
                # print("Incorrect: ",predictedEx.confidence, predictedEx.bow() )
                # print "Incorrect", '\t'.join([predictedEx.pred] + predictedEx.args) 
                sentence_y_scores.append(predictedEx.confidence)

            if(type(pickle_output) != type(None)):
                zero_conf_prec = sum(sentence_y_true) / len(predictedExtractions)
                zero_conf_recall = sum(sentence_y_true) / len(goldExtractions)

                (sentence_precision, sentence_recall, confidence_thresholds), optimal = Benchmark.prCurve(sentence_y_true.copy(), sentence_y_scores.copy(),
                                            recallMultiplier = ((len(goldExtractions) - sentence_unmatchedCount + 1)/float(len(goldExtractions))),
                                            sentence_level = True)
                sentence_precision = sentence_precision[:-1]
                sentence_recall = sentence_recall[:-1]

                ## Check with Samarth
                # Some systems have same scores for multiple extractions and sklearn computes precision and recall only for the unqiue scores
                # Fill in the values for duplicated scores
                num_predictions = len(sentence_y_scores)
                if(len(sentence_precision) < num_predictions):
                    sorted_y_scores = sorted(sentence_y_scores)
                    # Ensure that they are sorted in the same order - assuming increasing
                    assert sorted_y_scores[0] == confidence_thresholds[0]
                    assert sorted_y_scores[-1] == confidence_thresholds[-1]

                    sentence_precision_rep = np.zeros(num_predictions)
                    sentence_recall_rep = np.zeros(num_predictions)
                    for indx in range(num_predictions):
                        conf_indx = np.where(confidence_thresholds == sorted_y_scores[indx])[0]
                        # confidence_thresholds only contains unique values
                        assert len(conf_indx) == 1
                        conf_indx = conf_indx[0]
                        sentence_precision_rep[indx] = sentence_precision[conf_indx]
                        sentence_recall_rep[indx] = sentence_recall[conf_indx]
                    sentence_precision = sentence_precision_rep
                    sentence_recall = sentence_recall_rep

                assert(len(sentence_precision) == len(sentence_y_true)), ipdb.set_trace()

                # sorting p,r in ordering of decreasing confidence viz increasing recall
                sortedIndices = np.argsort(sentence_recall)
                sentence_precision = list(sentence_precision[sortedIndices])
                sentence_recall = list(sentence_recall[sortedIndices])
                predictedExtractions = sorted(predictedExtractions, key = lambda x: x.confidence , reverse = True)

                gold_extractions = ['('+ goldExtractions[i].args[0] + ';' + goldExtractions[i].pred + ';' + ";".join(goldExtractions[i].args[1:]) + ')' for i in range(len(goldExtractions))] 
                pred_extractions = ['('+ predictedExtractions[i].args[0] + ';' + predictedExtractions[i].pred + ';' + ";".join(predictedExtractions[i].args[1:]) + ')' for i in range(len(predictedExtractions)) ] #if predictedExtractions[i].confidence >= optimal[3]]

                pickle_output[goldSent] = {'gold':gold_extractions ,
                                     'pred':pred_extractions,
                                     'precisions':sentence_precision,
                                     'recalls':sentence_recall,
                                     'max_f1_score':optimal,
                                     'zero_conf_score':(round(zero_conf_prec,4), round(zero_conf_recall,4))
                                     }

            y_true += sentence_y_true
            y_scores += sentence_y_scores

        if(type(pickle_output_fp) != type(None) ):
            pickle.dump(pickle_output, open(pickle_output_fp, 'wb'))

        y_true = y_true
        y_scores = y_scores
        # print(correct, incorrect, unmatchedCount, correctTotal)
        # print("***Gold Extractions***")
        # print("\n".join([goldExappendtractions[i].pred + " ".join(goldExtractions[i].args) for i in range(6)]))
        # print("***Predicted Extappendractions***")
        # print("\n".join([predicappendtedExtractions[i].pred+ " ".join(predictedExtractions[i].args) for i in range(5)]))
        # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # to get to true recall we do:
        # r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        (p, r, confidence_thresholds), optimal = Benchmark.prCurve(np.array(y_true), np.array(y_scores),
                                            recallMultiplier = ((correctTotal - unmatchedCount)/float(correctTotal)))
        logging.info("AUC: {}\n Optimal (precision, recall, F1, threshold): {}".format(auc(r, p),
                                                                                       optimal))
        # Write error log to file
        if error_file:
            logging.info("Writing {} error indices to {}".format(len(errors),
                                                                 error_file))
            with open(error_file, 'w') as fout:
                fout.write('\n'.join([str(error)
                                     for error
                                      in errors]) + '\n')

        # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            # for cur_p, cur_r in sorted(zip(p, r), key = lambda (cur_p, cur_r): cur_r):
            for cur_p, cur_r in sorted(zip(p, r), key = lambda cur: cur[1]):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))

    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier, sentence_level = False):
        # Recall multiplier - accounts for the percentage examples unreached
        # Return (precision [list], recall[list]), (Optimal F1, Optimal threshold)

        # if this function is called at the sentence level, append a dummy point to y_true and y_scores so that precision and recall is calculated for all the confidence thresholds 
        if(sentence_level):
            y_true.append(1)
            y_true = np.array(y_true)
            y_scores.append(-1e32)
            y_scores = np.array(y_scores)

        y_scores = [score \
                    if not (np.isnan(score) or (not np.isfinite(score))) \
                    else 0
                    for score in y_scores]
        
        precision_ls, recall_ls, thresholds = precision_recall_curve(y_true, y_scores)
        recall_ls = recall_ls * recallMultiplier

        # removing the added point so that it does not affect the calculation of max_f1_confidence
        if(sentence_level):
            y_true = y_true[:-1]
            y_scores = y_scores[:-1]
            # removing the p=1,r=0 point from one end and the point that we introduced from the other end
            precision_ls = precision_ls[1:]
            recall_ls = recall_ls[1:]
            thresholds = thresholds[1:]

        optimal = max([(precision, recall, f_beta(precision, recall, beta = 1), threshold)
                       for ((precision, recall), threshold)
                       in zip(zip(precision_ls[:-1], recall_ls[:-1]),
                              thresholds)],
                      key = itemgetter(2))  # Sort by f1 score

        return ((precision_ls, recall_ls, thresholds),
                optimal)

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k).lower(), {'orig_sent': k, 'extractions': v}) for k, v in d.items()])

    @staticmethod
    def normalizeKey(k):
        # return Benchmark.removePunct(unicode(Benchmark.PTB_unescape(k.replace(' ','')), errors = 'ignore'))
        return Benchmark.removePunct(Benchmark.PTB_unescape(k.replace(' ','')))
        # return Benchmark.removePunct(Benchmark.PTB_unescape(k))
        # return k


    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s

    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s

    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)

    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Penn treebank bracket escapes
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'),]


def f_beta(precision, recall, beta = 1):
    """
    Get F_beta score from precision and recall.
    """
    beta = float(beta) # Make sure that results are in float
    return (1 + pow(beta, 2)) * (precision * recall) / ((pow(beta, 2) * precision) + recall)


f1 = lambda precision, recall: f_beta(precision, recall, beta = 1)




if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    logging.debug(args)

    if(args['--threshold']):
        threshold = float(args['--threshold'])
    else:
        threshold = None

    if args['--stanford']:
        predicted = StanfordReader()
        predicted.read(args['--stanford'])

    if args['--props']:
        predicted = PropSReader()
        predicted.read(args['--props'])

    if args['--ollie']:
        predicted = OllieReader()
        predicted.read(args['--ollie'])

    if args['--reverb']:
        predicted = ReVerbReader()
        predicted.read(args['--reverb'])

    if args['--clausie']:
        predicted = ClausieReader(threshold=threshold)
        predicted.read(args['--clausie'])

    if args['--openiefour']:
        predicted = OpenieFourReader()
        predicted.read(args['--openiefour'])

    if args['--openiefive']:
        predicted = OpenieFiveReader(threshold=threshold)
        predicted.read(args['--openiefive'])

    if args['--tabbed']:
        predicted = TabReader()
        predicted.read(args['--tabbed'])
 
    if args['--allennlp']:
        predicted = AllennlpReader(threshold=threshold)
        predicted.read(args['--allennlp'])

    # if args['--rnnoie']:
    #     predicted = RnnOIEReader(threshold=threshold)
    #     predicted.read(args['--rnnoie'])


    if args['--exactMatch']:
        matchingFunc = Matcher.argMatch

    elif args['--predMatch']:
        matchingFunc = Matcher.predMatch

    elif args['--argMatch']:
        matchingFunc = Matcher.argMatch

    else:
        matchingFunc = Matcher.lexicalMatch

    b = Benchmark(args['--gold'])
    out_filename = args['--out']

    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    b.compare(predicted = predicted.oie,
              matchingFunc = matchingFunc,
              output_fn = out_filename,
              error_file = args["--error-file"],
              pickle_output_fp=args['--pickle_output'])
