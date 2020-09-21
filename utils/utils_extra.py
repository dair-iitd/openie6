import os
import ipdb
import regex as re

# lines = []
# for line in open('carb/data/dev.txt.orig', 'r'):
#     line = line.strip('\n')
#     if len(set(line.split())) == 1:
#         lines += [line+' NONE \n']
#     else:
#         words = line.split()[:-2] + ['[unused1]', '[unused2]', '[unused3]']
#         lines += [' '.join(words)+'\n']

# open('carb/data/dev.txt', 'w').write(''.join(lines))

# for line in open('data/openie4_seq4.orig', 'r'):
#     line = line.strip('\n')
#     if '[unused1]' in line: # sentence
#         # fields = line.split('[unused1]')
#         # line = fields[0].strip() + ' [SEP] [unused1] ' + fields[1].strip()
#         lines += [line+'\n']
#     else: # labels
#         fields = line.split()
#         last_fields = fields[-3:]
#         last_fields = ['TYPE' if f == 'REL' else 'NONE' for f in last_fields]
#         # words = fields[:-3] + ['NONE'] + last_fields
#         words = fields[:-3] + last_fields
#         lines += [' '.join(words)+'\n']
# open('data/openie4_seq4', 'w').write(''.join(lines))

lines = []
total, arg1_mismatch, arg2_mismatch, rel_mismatch = 0, 0, 0, 0
for line in open('data/openie4_seq4', 'r'):
    total += 1
    line = line.strip('\n')
    if '[unused1]' in line: # sentence
        sentence = line
        lines += [line+'\n']
    else: # labels
        arg1_matches = re.findall('(ARG1 ?(ARG1 ?)*)', line)
        if len(arg1_matches) > 1:
            ipdb.set_trace()
            arg1_mismatch += 1
        arg2_matches = re.findall('(ARG2 ?(ARG2 ?)*)', line)    
        if len(arg2_matches) > 1:
            arg1_mismatch += 1
        rel_matches = re.findall('(REL ?(REL ?)*)', line)    
        if len(rel_matches) > 1:
            rel_mismatch += 1
        fields = line.split()
        # check non-consecutive arg1, arg2 and rel
        re.search('(ARG1 )*', line)
        lines += [' '.join(fields)+'\n']
print(arg1_mismatch, rel_mismatch, arg2_mismatch, total)

# open('data/openie4_seq4_iob', 'w').write(''.join(lines))
