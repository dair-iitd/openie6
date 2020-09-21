from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class ReVerbReader(OieReader):
    
    def __init__(self):
        self.inputSents = [sent.strip() for sent in open(ReVerbReader.RAW_SENTS_FILE).readlines()]
        self.name = 'ReVerb'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                arg1, rel, arg2 = data[2:5]
                confidence = data[11]
                text = self.inputSents[int(data[1])-1]
                
                curExtraction = Extraction(pred = rel, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
        
    # ReVerb requires a different files from which to get the input sentences
    # Relative to repo root folder
    RAW_SENTS_FILE = './raw_sentences/all.txt'    


