from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class OllieReader(OieReader):
    
    def __init__(self):
        self.name = 'OLLIE'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            fin.readline() #remove header
            for line in fin:
                data = line.strip().split('\t')
                confidence, arg1, rel, arg2, enabler, attribution, text  = data[:7]
                curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
    

