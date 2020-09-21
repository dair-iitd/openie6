from .oieReader import OieReader
from .extraction import Extraction
from _collections import defaultdict
import ipdb

class GoldReader(OieReader):
    
    # Path relative to repo root folder
    default_filename = './oie_corpus/all.oie' 
    
    def __init__(self):
        self.name = 'Gold'
    
    def read(self, fn):
        d = defaultdict(lambda: [])
        with open(fn) as fin:
            for line_ind, line in enumerate(fin):
#                print line
                data = line.strip().split('\t')
                text, rel = data[:2]
                args = data[2:]
                confidence = 1
                
                curExtraction = Extraction(pred = rel.strip(),
                                           head_pred_index = None,
                                           sent = text.strip(),
                                           confidence = float(confidence),
                                           index = line_ind)
                for arg in args:
                    if "C: " in arg:
                        continue
                    arg = arg.replace('L:','')
                    arg = arg.replace('T:','')
                    curExtraction.addArg(arg.strip())
                    
                d[text.strip()].append(curExtraction)
        self.oie = d
        

if __name__ == '__main__' :
    g = GoldReader()
    g.read('../oie_corpus/all.oie', includeNominal = False)
    d = g.oie
    e = d.items()[0]
    print(e[1][0].bow())
    print(g.count())
