import ipdb
import json
import re
import argparse

CARB_TEST_FILE = 'carb/data/dev_gold_allennlp_format.txt'


def load_WiRe_annotations():
    save_path = "../data/WiRe57_343-manual-oie.json"
    annotations = json.load(open(save_path))
    return annotations


def get_extraction_wire57(arg1, rel, arg2):
    return {'arg1': arg1, 'rel': rel, 'arg2': arg2}


def get_extraction_wire57_gold(arg1, rel, arg2):
    extraction = {}
    extraction['arg1'] = {'text': arg1, 'words': arg1.split()}
    extraction['rel'] = {'text': rel, 'words': rel.split()}
    extraction['arg2'] = {'text': arg2, 'words': arg2.split()}
    return extraction


def get_allenlp_args(line):
    assert len(re.findall("<arg1>.*</arg1>", line)) == 1
    assert len(re.findall("<rel>.*</rel>", line)) == 1
    assert len(re.findall("<arg2>.*</arg2>", line)) == 1

    arg1 = re.findall("<arg1>.*</arg1>", line)[0].strip('<arg1>').strip('</arg1>').strip()
    rel = re.findall("<rel>.*</rel>", line)[0].strip('<rel>').strip('</rel>').strip()
    arg2 = re.findall("<arg2>.*</arg2>", line)[0].strip('<arg2>').strip('</arg2>').strip()

    return arg1, rel, arg2


def process_allennlp_format(file, gold=False):
    with open(file, 'r') as f:
        lines = f.readlines()

    extractions = {}

    current_sentence = None
    for l in lines:
        if len(l.strip()) > 0:
            items = l.strip().split('\t')
            assert len(items) == 3
            if current_sentence != items[0]:
                current_sentence = items[0]
                extractions[current_sentence] = []
            arg1, rel, arg2 = get_allenlp_args(items[1])
            if gold:
                extr = get_extraction_wire57_gold(arg1, rel, arg2)
            else:
                extr = get_extraction_wire57(arg1, rel, arg2)
            extractions[current_sentence].append(extr)

    return extractions


def main(arguments):

    gold = process_allennlp_format(arguments.gold, gold=True)

    predictions_by_OIE = process_allennlp_format(arguments.system)

    report = ""
    metrics, raw_match_scores = eval_system(gold, predictions_by_OIE)

    prec, rec = metrics['precision'], metrics['recall']
    f1_score = f1(prec, rec)

    report += ("prec/rec/f1: {:.1%} {:.1%} {:.3f}".format(prec, rec, f1_score))

    print(report)

def eval_system(gold, predictions):
    results = {}
    # Get a manytuples-to-manytuples match-score for each sentence,
    # then gather the scores across sentences and compute the weighted-average
    for s, reference_tuples in gold.items():
        predicted_tuples = predictions.get(s, [])
        results[s] = sentence_match(reference_tuples, predicted_tuples)

    prec_num, prec_denom = 0, 0
    rec_num, rec_denom = 0, 0

    for s in results.values():
        prec_num += s['precision'][0]
        prec_denom += s['precision'][1]
        rec_num += s['recall'][0]
        rec_denom += s['recall'][1]

    metrics = {
        'precision': prec_num / prec_denom,
        'recall': rec_num / rec_denom,
    }
    raw_match_scores = None
    return metrics, raw_match_scores


# TODO:
# - Implement half points for part-misplaced words.
# - Deal with prepositions possibly being the first token of an arg, especially for arg2.
#   > It's fully ok for "any" prep to be last word of ref_rel or first_word of pred_arg


def avg(l):
    return sum(l) / len(l)


def f1(prec, rec):
    try:
        return 2 * prec * rec / (prec + rec)
    except ZeroDivisionError:
        return 0


def sentence_match(gold, predicted):
    """For a given sentence, compute tuple-tuple matching scores, and gather them
at the sentence level. Return scoring metrics."""
    score, maximum_score = 0, len(gold)
    exact_match_scores = [[None for _ in predicted] for __ in gold]
    scores = [[None for _ in predicted] for __ in gold]
    for i, gt in enumerate(gold):
        for j, pt in enumerate(predicted):
            exact_match_scores[i][j] = tuple_exact_match(pt, gt)
            scores[i][j] = tuple_match(pt, gt)  # this is a pair [prec,rec] or False
    scoring_metrics = aggregate_scores_greedily(scores)

    return scoring_metrics


def str_list(thing):
    return "\n".join([str(s) for s in thing])


def aggregate_scores_greedily(scores):
    # Greedy match: pick the prediction/gold match with the best f1 and exclude
    # them both, until nothing left matches. Each input square is a [prec, rec]
    # pair. Returns precision and recall as score-and-denominator pairs.
    matches = []
    while True:
        max_s = 0
        gold, pred = None, None
        for i, gold_ss in enumerate(scores):
            if i in [m[0] for m in matches]:
                # Those are already taken rows
                continue
            for j, pred_s in enumerate(scores[i]):
                if j in [m[1] for m in matches]:
                    # Those are used columns
                    continue
                if pred_s and f1(*pred_s) > max_s:
                    max_s = f1(*pred_s)
                    gold = i
                    pred = j
        if max_s == 0:
            break
        matches.append([gold, pred])
    # Now that matches are determined, compute final scores.
    prec_scores = [scores[i][j][0] for i, j in matches]
    rec_scores = [scores[i][j][1] for i, j in matches]
    total_prec = sum(prec_scores)
    total_rec = sum(rec_scores)
    scoring_metrics = {"precision": [total_prec, len(scores[0])],
                       "recall": [total_rec, len(scores)],
                       "precision_of_matches": prec_scores,
                       "recall_of_matches": rec_scores
                       }
    # print(scoring_metrics)
    return scoring_metrics


def aggregate_exact_matches(match_matrix):
    # For this agregation task, no predicted tuple can exact-match two gold
    # ones, so it's easy, look at lines and columns looking for OR-total booleans.
    recall = [sum([any(gold_matches) for gold_matches in match_matrix], 0), len(match_matrix)]
    # ^ this is [3,5] for "3 out of 5", to be lumped together later.
    if len(match_matrix[0]) == 0:
        precision = [0, 0]  # N/A
    else:
        precision = [sum([any([g[i] for g in match_matrix]) for i in range(len(match_matrix[0]))], 0),
                     len(match_matrix[0])]
    # f1 = 2 * precision * recall / (precision + recall)
    metrics = {'precision': precision,
               'recall': recall}
    return metrics


def part_to_string(p):
    return " ".join(p['words'])


def gold_to_text(gt):
    text = " ; ".join([part_to_string(gt['arg1']), part_to_string(gt['rel']), part_to_string(gt['arg2'])])
    if gt['arg3+']:
        text += " ; " + " ; ".join(gt['arg3+'])
    return text


def tuple_exact_match(t, gt):
    """Without resolving coref and WITH the need to hallucinate humanly infered
words, does the tuple match the reference ? Returns a boolean."""
    for part in ['arg1', 'rel', 'arg2']:
        if not t[part] == ' '.join(gt[part]['words']):
            # This purposedly ignores that some of the gt words are 'inf'
            # print("Predicted '{}' is different from reference '{}'".format(t[part], ' '.join(gt[part]['words'])))
            return False
    return True


"""
Wire57 tuples are built like so:
t = {"attrib/spec?" : attrib,
     "arg1" : {'text' : arg1, 'words': arg1_w, "words_indexes" : arg1_ind,
               'dc_text' : arg1dc, 'decorefed_words' : arg1dc_w, 'decorefed_indexes' : arg1dc_ind},
     "rel" : {'text' : rel, 'words': rel_w, "words_indexes" : rel_ind},
     "arg2" : {'text' : arg2, 'words': arg2_w, "words_indexes" : arg2_ind,
               'dc_text' : arg2dc, 'decorefed_words' : arg2dc_w, 'decorefed_indexes' : arg2dc_ind},

"""


def tuple_match(t, gt):
    """t is a predicted tuple, gt is the gold tuple. How well do they match ?
Yields precision and recall scores, a pair of non-zero values, if it's a match, and False if it's not.
    """
    precision = [0, 0]  # 0 out of 0 predicted words match
    recall = [0, 0]  # 0 out of 0 reference words match
    # If, for each part, any word is the same as a reference word, then it's a match.
    for part in ['arg1', 'rel', 'arg2']:
        predicted_words = t[part].split()
        gold_words = gt[part]['words']
        if not predicted_words:
            if gold_words:
                return False
            else:
                continue
        matching_words = sum(1 for w in predicted_words if w in gold_words)
        if matching_words == 0:
            return False  # t <-> gt is not a match
        precision[0] += matching_words
        precision[1] += len(predicted_words)
        # Currently this slightly penalises systems when the reference
        # reformulates the sentence words, because the reformulation doesn't
        # match the predicted word. It's a one-wrong-word penalty to precision,
        # to all systems that correctly extracted the reformulated word.
        recall[0] += matching_words
        recall[1] += len(gold_words)

    if recall[1] == 0 or precision[1] == 0:
        return False
    prec = precision[0] / precision[1]
    rec = recall[0] / recall[1]
    return [prec, rec]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', help="file path for gold in allennlp format", required=True)
    parser.add_argument('--system', help="file path for system in allennlp format", required=True)
    arguments = parser.parse_args()
    main(arguments)

