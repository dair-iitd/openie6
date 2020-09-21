from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class StanfordReader(OieReader):
    
    def __init__(self):
        self.name = 'Stanford'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                arg1, rel, arg2 = data[2:5]
                confidence = data[11]
                text = data[12]
                
                curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
