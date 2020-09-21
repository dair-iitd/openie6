import ipdb
import json
import argparse
import rouge
from rouge import rouge_n_sentence_level
import math
import thinqpbo as tq
from tqdm import tqdm

node_t = 0.65
edge_t = 0.85

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fp')
    parser.add_argument('--out_fp')

    return parser

def qpbo(out_fp, sent_list, graphD, node_edgeD):
    # Create graph object.
    # Number of nodes to add.
    f = open(out_fp, "w")
    cnt = 0
    for i in tqdm(range(len(sent_list))):
        graph = tq.QPBODouble()
        nodes_to_add = len(graphD[sent_list[i]])
        num = nodes_to_add
        first_node_id = graph.add_node(nodes_to_add)
        visit = dict()
        for j in range(num):
            node_score = node_edgeD[sent_list[i]][j]
            if node_score < node_t:
                visit[j] = False
                continue
            visit[j] = True
            graph.add_unary_term(j, 0, -node_edgeD[sent_list[i]][j])
        l1 = list(visit.values())
        for j in range(num):
            for k in range(j+1, num):
                edge_score = node_edgeD[sent_list[i]][(j,k)]
                if not((visit[j] == True) and (visit[k] == True) and (edge_score > edge_t)):
                    continue
                graph.add_pairwise_term(j, k, 0, 0, 0, edge_score)      
        graph.solve()
        graph.compute_weak_persistencies()
        twice_energy = graph.compute_twice_energy()
        for n in range(nodes_to_add):
            cnt += 1
            segment = graph.get_label(n)
            if segment == 1:
                f.write(sent_list[i] + '\t' + graphD[sent_list[i]][n] + '\t' + str(math.log(node_edgeD[sent_list[i]][n])) + '\n')
    # Add two nodes.
    # Add edges.
    print(cnt)
    return  

def get_data(inp_fp):
    inp_f = open(inp_fp,'r')
    extD = dict()
    graphD = dict()
    node_edgeD = dict()
    for line in inp_f:
        line = line.strip('\n')
        sentence, extraction, confidence = line.split('\t')
        if sentence not in extD:
            extD[sentence] = list()

        already_added = False
        for added_extraction, _ in extD[sentence]:
            if extraction == added_extraction:
                already_added = True
        if already_added:
            continue

        extD[sentence].append([extraction, confidence])
    for key in extD.keys():
        graphD[key] = dict()
        cnt = 0
        for item in extD[key]:
            graphD[key][cnt]=item[0]
            cnt += 1
    sent_list = []
    sent_dict = extD
    for key in tqdm(sent_dict):
        sent_list.append(key)
        node_edgeD[key] = dict()
        num = len(sent_dict[key])
        key_sum = 0
        for i in range(num):
            key_sum += math.exp(float(extD[key][i][1]))

        for i in range(num):
            node_edgeD[key][i] = (math.exp(float(extD[key][i][1])))
                    
        edge_sum = 0
        for i in range(0,num):
            for j in range(i+1, num):
                sent1 = ''.join(sent_dict[key][i])
                sent2 = ''.join(sent_dict[key][j])
                
                recall, precision, rouge = rouge_n_sentence_level(sent1, sent2, 2)
                edge_sum += rouge
        for i in range(0,num):
            for j in range(i+1, num):
                sent1 = ''.join(sent_dict[key][i])
                sent2 = ''.join(sent_dict[key][j])
                recall, precision, rouge = rouge_n_sentence_level(sent1, sent2, 2)

                node_edgeD[key][(i,j)] = rouge
    return sent_list, extD, graphD, node_edgeD
    

def main():
    parser = parse_args()
    args = parser.parse_args()
    sent_list, extD, graphD, node_edgeD = get_data(args.inp_fp)
    qpbo(args.out_fp, sent_list, graphD, node_edgeD)

if __name__ == '__main__':
    main()  

