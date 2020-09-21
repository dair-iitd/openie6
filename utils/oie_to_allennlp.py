import argparse
from tqdm import tqdm
import regex as re
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--conj', type=str) # original\n(split sentences) file for mapping

    return parser

parser = parse_args()
args = parser.parse_args()
    
in_f = open(args.inp,'r')
content = in_f.read()
examples = content.split('\n\n')

conj_mapping = dict()
conj_mapping_values = set()
if args.conj:
    content = open(args.conj).read()
    for example in content.split('\n\n'):
        for i, line in enumerate(example.strip('\n').split('\n')):
            if i == 0:
                orig_sentence = line
            else:
                conj_mapping[line] = orig_sentence
    conj_mapping_values = conj_mapping.values()

out_f = open(args.out,'w')
for example_id, example in tqdm(enumerate(examples)):
    lines = example.split('\n')
    sentence = lines[0]
    if sentence in conj_mapping_values: # ignore extractions of original sentence
        continue
    if sentence in conj_mapping: # replace split sentence with original sentence
        sentence = conj_mapping[sentence]

    extractions = lines[1:]
    for extraction in extractions:
        confidence = extraction.split(' ')[0].strip(':')
        extraction = ' '.join(extraction.split(' ')[1:])
        if 'Context' in extraction:
            extraction = ' '.join(extraction.split(':')[1:])
        fields = extraction.split(';')
        # try:
        #     match = re.search('(.*):\((.*); (.*); (.*)\)', extraction)
        #     # includes context
        #     confidence = match.group(1).strip()
        #     confidence = confidence.split()[0]
        # except:
        #     match = re.search('(\d.\d\d) \((.*); (.*); (.*)\)', extraction)
        #     confidence = match.group(1).strip()
        
        subject = fields[0][1:].strip() # remove opening bracket
        relation = fields[1].strip()
        object = ' '.join(fields[2:])[:-1].strip() # remove closing bracket
        object = object.replace('L:','')
        object = object.replace('T:','')
        out_f.write(f'{sentence}\t<arg1> {subject} </arg1> <rel> {relation} </rel> <arg2> {object} </arg2>\t{confidence}\n')        
out_f.close()
