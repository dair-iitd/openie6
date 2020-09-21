from .oieReader import OieReader
from .extraction import Extraction
import math
import os
import ipdb

class AllennlpReader(OieReader):
    
    def __init__(self, threshold):
        self.name = 'Allennlp'
        self.threshold = threshold
    
    def read(self, fn):
        d = {}
        # with open(fn) as fin:
        if os.path.exists(fn):
            # print("reading from file")
            fin = open(fn)
        else:
            # print("reading from string")
            fin = fn.strip().split('\n')
        
        for line in fin:
            '''
            data = line.strip().split('\t')
            confidence = data[0]
            if not all(data[2:5]):
                continue
            arg1, rel = [s[s.index('(') + 1:s.index(',List(')] for s in data[2:4]]
            #args = data[4].strip().split(');')
            #print arg2s
            args = [s[s.index('(') + 1:s.index(',List(')] for s in data[4].strip().split(');')]
            # if arg1 == "the younger La Flesche":
            #    print len(args)
            text = data[5]
            if data[1]:
                #print arg1, rel
                s = data[1]
                if not (arg1 + ' ' + rel).startswith(s[s.index('(') + 1:s.index(',List(')]):
                    #print "##########Not adding context" 
                    arg1 = s[s.index('(') + 1:s.index(',List(')] + ' ' + arg1
                    #print arg1 + rel, ",,,,, ", s[s.index('(') + 1:s.index(',List(')] 
            '''
            #print(line)
            line = line.strip().split('\t')
            #print(line)
            text = line[0]
            try:
                confidence = line[2]
            except:
                raise Exception('Unable to find confidence in line: ',line)
            line = line[1]
            try:
                arg1 = line[line.index('<arg1>') + 6:line.index('</arg1>')]
            except:
                arg1 = ""
            try:
                rel = line[line.index('<rel>') + 5:line.index('</rel>')]
            except:
                rel = ""
            try:
                arg2 = line[line.index('<arg2>') + 6:line.index('</arg2>')]
            except:
                arg2 = ""

            if(type(self.threshold) != type(None) and float(confidence) < self.threshold):
                continue

            if not ( arg1 or arg2 or rel):
                continue
            #confidence = 1
            #print(arg1, rel, arg2, confidence)
            # curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = -1/float(confidence))
            # curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = math.exp(float(confidence)))
            curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = float(confidence))
            curExtraction.addArg(arg1)
            curExtraction.addArg(arg2)
            #for arg in args:
            #    curExtraction.addArg(arg)
            d[text] = d.get(text, []) + [curExtraction]
        
        if os.path.exists(fn):
            fin.close()

        self.oie = d
