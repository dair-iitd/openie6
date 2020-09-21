# This script prints the outputs of the OIE systems in human readable format:
# Sentence
# confidence1 extraction1
# confidence2 extraction2
# confidence3 extraction3
# <Empty line>


import sys
import argparse
import ipdb
import os
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default='allennlp')
    parser.add_argument("--inp_fp")
    parser.add_argument("--out_fp")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--system", action='append', type=str)

    return parser

def props_reader(input_filename, output_filename, threshold):
    f=open(input_filename, 'r')
    lines=f.readlines()
    f.close()

    extractions=dict()
    for i in range(len(lines)):
        if not lines[i].strip():
            continue
        data = lines[i].strip().split('\t')
        confidence, text, rel = data[:3]
        confidence = round( float(confidence),4)
        if threshold != None and confidence < threshold:
            continue

        line = "("
        if len(data)>=4:
                line += data[4] + " ; "
        line += rel
        for arg in data[6::2]:
                line += " ; " + arg
        line += ")"

        if text not in extractions.keys():
            extractions[text] = []
        extractions[text].append((confidence, line))

    for sent in extractions.keys():
        print(sent)
        for tup in extractions[sent]:
            print(tup[0] , tup[1])
        print()
        # break

def clausie_reader(input_filename, output_filename, threshold):
    f=open(input_filename, 'r')
    lines=f.readlines()
    f.close()

    extractions=dict()
    for i in range(len(lines)):
        data = lines[i].strip().split('\t')
        if len(data) == 1:
            sentence = data[0]
            extractions[sentence] = []
        elif len(data) == 5:
            arg1, rel, arg2 = [s[1:-1] for s in data[1:4]]
            confidence = round( float(data[4]), 4)
            if threshold != None and confidence < threshold:
                continue
            line = "("+ arg1 + " ; " + rel + " ; " + arg2 + ")"
            extractions[sentence].append((confidence, line))

    for sent in extractions.keys():
        print(sent)
        for tup in extractions[sent]:
            print(tup[0] , tup[1])
        print()
            # break

def allennlp_reader(input_filename, output_filename, threshold, systems):
    if systems != None:
        all_lines = []
        for system in systems:
            lines = open('data/train/{}/extractions.tsv'.format(system),'r').readlines()
            lines = [' '.join(line.split('\t')[1].split()) for line in lines]
            all_lines.append(set(lines))


    f=open(input_filename, 'r')
    lines=f.readlines()
    f.close()

    out_f = open(output_filename, 'w')

    extractions=dict()
    for i in range(len(lines)):
        line = lines[i].split('\t')
        confidence = float(line[2])
        if threshold != None and confidence < threshold:
            continue
        found_systems = []
        if systems != None:
            for system_id in range(len(systems)):
                if ' '.join(line[1].split()) in all_lines[system_id]:
                    found_systems.append(systems[system_id])
            # if len(found_systems) == 0:
            #     ipdb.set_trace()

        if line[0] in extractions:
            extractions[line[0]].append((line[1], round(float(line[2]),4), found_systems))
        else:
            extractions[line[0]] = [(line[1], round(float(line[2]),4), found_systems)]

    for sent in extractions.keys():
        out_f.write(sent+'\n')
        extractions[sent] = sorted(extractions[sent], key=lambda x:x[1], reverse=True)
        for tup in extractions[sent]:
            pred=tup[0]
            pred=pred.replace("<arg1>","(")
            pred=pred.replace("</arg1> <rel>",";")
            pred=pred.replace("</rel> <arg2>",";")
            pred=pred.replace("</arg2>",")")
            if systems != None:
                out_f.write(str(tup[1])+' '+pred+' '+','.join(tup[2])+'\n')
            else:
                out_f.write(str(math.exp(tup[1]))+' '+pred+'\n')
        out_f.write('\n')
        # break

def openiefive_reader(input_filename, output_filename, threshold):
    f=open(input_filename, 'r')
    lines=f.readlines()
    f.close()

    extractions=dict()
    for i in range(len(lines)):
        # print(lines[i])
        line = lines[i].split('\t')
        sentence = line[-1].strip()
        confidence = round(float(line[0]) , 4)
        if threshold != None and confidence < threshold:
            continue
        line = '\t'.join(line[1:-1]).strip().split('\t')

        arg1_done = False
        context = None
        arg1 = None
        rel = None
        arg2 = None
        location = None
        time = None
        for j in range(len(line)):
            if line[j][:14]=="SimpleArgument" and arg1_done==False:
                arg1 = line[j][ line[j].index('(') + 1 : line[j].index(",List") ]
                arg1_done = True

            elif line[j][:14]=="SimpleArgument" and arg1_done==True:
                arg2 = line[j][ line[j].index('(') + 1 : line[j].index(",List") ]

            elif line[j][:8]=="Relation":
                rel = line[j][ line[j].index('(') + 1 : line[j].index(",List") ]

            elif line[j][:15]=="SpatialArgument":
                location = line[j][ line[j].index('(') + 1 : line[j].index(",List") ]

            elif line[j][:16]=="TemporalArgument":
                time = line[j][ line[j].index('(') + 1 : line[j].index(",List") ]

            elif line[j][:7]=="Context":
                context = line[j][ line[j].index('(') + 1 : line[j].index(",List") ]

            else:
                raise Exception("argument not matching" , j)

        pred = "(" 
        if context!=None:
            pred += " Context : " + context + " ; "

        pred += arg1 + " ; " + rel
        
        if arg2!=None:
            pred += " ; " + arg2 
        if location!=None:
            pred += " ; L : " + location
        if time!=None:
            pred += " ; T : " + time
        
        pred += ")"
        # print(line)
        # print(pred, "\n")
        # break

        if sentence in extractions.keys():
            extractions[sentence].append((pred,confidence))
        else:
            extractions[sentence] = [(pred ,confidence)]

    for sent in extractions.keys():
        print(sent)
        for tup in extractions[sent]:
            print(tup[1] , tup[0])
        print()

def gold_reader(input_filename, output_filename):
    f=open(input_filename, 'r')
    lines=f.readlines()
    f.close()

    out_f=open(output_filename, 'w')
    extractions=dict()
    for i in range(len(lines)):
        # print(lines[i])
        line = lines[i].split('\t')
        sentence = line[0].strip()
        rel = line[1].strip()
        arg1 = line[2].strip()
        arg2_and_beyond = line[3:]

        pred = "(" 
        pred += arg1 + " ; " + rel
        
        for arg in arg2_and_beyond:
            arg = arg.strip()
            if(arg != ""):
                pred += " ; " + arg
        pred += ")"

        if sentence in extractions.keys():
            extractions[sentence].append(pred)
        else:
            extractions[sentence] = [pred]

    for sent in extractions.keys():
        out_f.write(sent+'\n')
        # print(sent)
        for tup in extractions[sent]:
            out_f.write('1 '+tup+'\n')
            # print(tup)
        out_f.write('\n')
        # print()

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    if args.out_fp == None:
        if args.inp_fp.endswith('tsv'):
            args.out_fp = os.path.splitext(args.inp_fp)[0] + '.txt'
        else:
            args.out_fp = args.inp_fp + '.txt'
    print('writing to ',args.out_fp,' ...')

    if args.format =='gold':
        gold_reader(args.inp_fp, args.out_fp)
    
    elif args.format =='allennlp':
        allennlp_reader(args.inp_fp, args.out_fp, args.threshold, args.system)

    elif args.format =='openiefive':
        openiefive_reader(args.inp_fp, args.out_fp, args.threshold)

    elif args.format =='clausie':
        clausie_reader(args.inp_fp, args.out_fp, args.threshold)

    elif args.format =='props':
        props_reader(args.inp_fp, args.out_fp, args.threshold)
