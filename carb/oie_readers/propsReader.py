from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction


class PropSReader(OieReader):
    
    def __init__(self):
        self.name = 'PropS'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                if not line.strip():
                    continue
                data = line.strip().split('\t')
                confidence, text, rel = data[:3]
                curExtraction = Extraction(pred = rel, sent = text, confidence = float(confidence), head_pred_index=-1)
                
                for arg in data[4::2]:
                    curExtraction.addArg(arg)
                    
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
        # self.normalizeConfidence()
    
    
    def normalizeConfidence(self):
        ''' Normalize confidence to resemble probabilities '''        
        EPSILON = 1e-3
        
        self.confidences = [extraction.confidence for sent in self.oie for extraction in self.oie[sent]]
        maxConfidence = max(self.confidences)
        minConfidence = min(self.confidences)
        
        denom = maxConfidence - minConfidence + (2*EPSILON)
        
        for sent, extractions in self.oie.items():
            for extraction in extractions:
                extraction.confidence = ( (extraction.confidence - minConfidence) + EPSILON) / denom

    
    
    
