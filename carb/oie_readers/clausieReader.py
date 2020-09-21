""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

Convert to tabbed format
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt

# Local imports
from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction
import ipdb
#=-----

class ClausieReader(OieReader):
    
    def __init__(self):
        self.name = 'ClausIE'
    
    def read(self, fn):
        d = {}
        with open(fn, encoding="utf-8") as fin:
            for line in fin:
                data = line.strip().split('\t')
                if len(data) == 1:
                    text = data[0]
                elif len(data) == 5:
                    arg1, rel, arg2 = [s[1:-1] for s in data[1:4]]
                    confidence = data[4]
                    
                    curExtraction = Extraction(pred = rel,
                                               head_pred_index = -1,
                                               sent = text,
                                               confidence = float(confidence))

                    curExtraction.addArg(arg1)
                    curExtraction.addArg(arg2)
                    d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
        # self.normalizeConfidence()

        # # remove exxtractions below the confidence threshold
        # if type(self.threshold) != type(None):
        #     new_d = {}
        #     for sent in self.oie:
        #         for extraction in self.oie[sent]:
        #             if extraction.confidence < self.threshold:
        #                 continue
        #             else:
        #                 new_d[sent] = new_d.get(sent, []) + [extraction]
        #     self.oie = new_d

    
    
    def normalizeConfidence(self):
        ''' Normalize confidence to resemble probabilities '''        
        EPSILON = 1e-3
        
        confidences = [extraction.confidence for sent in self.oie for extraction in self.oie[sent]]
        maxConfidence = max(confidences)
        minConfidence = min(confidences)
        
        denom = maxConfidence - minConfidence + (2*EPSILON)
        
        for sent, extractions in self.oie.items():
            for extraction in extractions:
                extraction.confidence = ( (extraction.confidence - minConfidence) + EPSILON) / denom



if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)


    oie = ClausieReader()
    oie.read(inp_fn)
    oie.output_tabbed(out_fn)

    logging.info("DONE")
