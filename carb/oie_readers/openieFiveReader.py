from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class OpenieFiveReader(OieReader):

    def __init__(self):
        self.name = 'OpenIE-5'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                confidence = data[0]

                if not all(data[2:5]):
                    continue
                arg1, rel = [s[s.index('(') + 1:s.index(',List(')] for s in data[2:4]]
                #args = data[4].strip().split(');')
                #print arg2s
                args = [s[s.index('(') + 1:s.index(',List(')] for s in data[4].strip().split(');')]
#                if arg1 == "the younger La Flesche":
#                    print len(args)
                text = data[5]
                if data[1]:
                    #print arg1, rel
                    s = data[1]
                    if not (arg1 + ' ' + rel).startswith(s[s.index('(') + 1:s.index(',List(')]):
                        #print "##########Not adding context" 
                        arg1 = s[s.index('(') + 1:s.index(',List(')] + ' ' + arg1
                        #print arg1 + rel, ",,,,, ", s[s.index('(') + 1:s.index(',List(')] 
                #curExtraction = Extraction(pred = rel, sent = text, confidence = float(confidence))
                curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                for arg in args:
                    curExtraction.addArg(arg)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
