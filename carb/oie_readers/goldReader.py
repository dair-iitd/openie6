from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction
from _collections import defaultdict
import ipdb

class GoldReader(OieReader):
    
    # Path relative to repo root folder
    default_filename = './oie_corpus/all.oie' 
    
    def __init__(self):
        self.name = 'Gold'
    
    def read(self, fn):
        # d = defaultdict(lambda: [])
        d = dict()
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
                    curExtraction.addArg(arg.strip())

                if text.strip() not in d:   
                    d[text.strip()] = []
                d[text.strip()].append(curExtraction)
        self.oie = d
        

if __name__ == '__main__' :
    g = GoldReader()
    g.read('../oie_corpus/all.oie', includeNominal = False)
    d = g.oie
    e = d.items()[0]
    print(e[1][0].bow())
    print(g.count())
