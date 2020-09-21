import sys
sys.path.insert(0, 'carb')

import metric
import data

import os
import pickle
import argparse
import sys
import ipdb

sys.path.insert(0, '/home/keshav/teranishi/coordparser/src')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_fp', type=str, required=True)
    parser.add_argument('--tera_fp', type=str, required=True)
    return parser

def parse_labels(inp_fp):
    inp = pickle.load(open(inp_fp,'rb'))
    all_predictions, all_gt, all_sentences = [], [], []
    for ex_idx, example in enumerate(inp):
        predict_coords = metric.get_coords(example[1])
        gt_coords = metric.get_coords(example[2])
        all_predictions.append(predict_coords)
        all_gt.append(gt_coords)
        all_sentences.append(example[0])
        
    return all_sentences, all_predictions, all_gt

def parse_teranishi(inp_fp):
    all_inps = pickle.load(open(inp_fp,'rb'))
    all_sents = [inp[0] for inp in all_inps]
    all_coords = [inp[1] for inp in all_inps]

    return all_sents, all_coords

def compare_coords(pred_coord, gt_coord, label):
    if gt_coord == None and pred_coord == None:
        return True
    if (gt_coord != None and pred_coord == None) or (gt_coord == None and pred_coord != None):
        return False
    if gt_coord.conjuncts == pred_coord.conjuncts:
        if label and gt_coord.label == pred_coord.label:
            return True
        else:
            return False
    else:
        return False

def check_constraints(sentences, label_coords, tera_coords, gt_coords):
    num_examples = len(label_coords)
    assert len(tera_coords) == num_examples
    assert len(gt_coords) == num_examples

    labels_acc, labels_total, tera_acc, total = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    wrong_conjuncts, wrong_depth, upper_found, wrong_depth_correct_conjuncts = 0, 0, 0, 0
    # not_found, none, total1 = 0, 0, 0
    for ex_indx in range(num_examples):
        ex_label_coords = label_coords[ex_indx]
        ex_tera_coords = tera_coords[ex_indx]
        ex_gt_coords = gt_coords[ex_indx]
        words = sentences[ex_indx].split()
        for coord_index in gt_coords[ex_indx]:
            label_coord = ex_label_coords.get(coord_index, None)
            tera_coord = ex_tera_coords.get(coord_index, None)
            gt_coord = ex_gt_coords[coord_index]

            depth = gt_coord.label if gt_coord != None else 3
            labels_acc[depth] += compare_coords(label_coord, gt_coord, label=True)
            if label_coord != None:
                labels_total[label_coord.label] += 1
            else:
                labels_total[3] += 1
            tera_acc[depth] += compare_coords(tera_coord, gt_coord, label=False)
            total[depth] += 1

            if depth == 1:
                if compare_coords(label_coord, gt_coord, label=True) == 0:
                    if label_coord != None:
                        if label_coord.label != 1:
                            wrong_depth += 1
                            if compare_coords(label_coord, gt_coord, label=False) == 1:
                                wrong_depth_correct_conjuncts += 1
                        else:
                            wrong_conjuncts += 1
                    else:
                        for coord in ex_label_coords.values():
                            if coord == None:
                                continue
                            if coord.label != 0:
                                continue
                            if coord_index in range(coord.conjuncts[0][0], coord.conjuncts[-1][1]):
                                upper_found += 1

            # if depth == 0 and gt_coord != None:
            #     start_word_idx, end_word_idx = gt_coord.conjuncts[0][0], gt_coord.conjuncts[-1][1]
            #     conjunction_words = ' '.join(words[start_word_idx:end_word_idx+1])
            #     for word_idx in range(start_word_idx, end_word_idx+1):
            #         if word_idx == coord_index:
            #             continue
            #         if words[word_idx] in data.CC_KEY:
            #             total1 += 1
            #             if word_idx in ex_gt_coords and word_idx not in ex_label_coords:
            #                 not_found += 1
                        # if word_idx not in ex_gt_coords:
                        #     not_found += 1
                        #     continue
                        # if ex_gt_coords[word_idx] == None:
                        #     none += 1
                # for other_index in ex_gt_coords:
                #     if ex_gt_coords[other_index].label == 2:
                #         if coord_index>=ex_gt_coords[other_index].conjuncts[0][0] and \
                #          coord_index<=ex_gt_coords[other_index].conjuncts[-1][1]:
                #             higher_index = other_index
    print('Finished all examples...')
    ipdb.set_trace()


def main():
    parser = parse_args()
    args = parser.parse_args()

    labels_sentences, labels_predictions, gt = parse_labels(args.labels_fp)
    tera_sentences, tera_predictions = parse_teranishi(args.tera_fp)

    assert labels_sentences == tera_sentences
    check_constraints(labels_sentences, labels_predictions, tera_predictions, gt)

if __name__ == '__main__':
    main()