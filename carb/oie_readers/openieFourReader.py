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

class OpenieFourReader(OieReader):

    def __init__(self):
        self.name = 'OpenIE-4'

    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                confidence = data[0]
                if not all(data[2:5]):
                    logging.debug("Skipped line: {}".format(line))
                    continue
                arg1, rel, arg2 = [s[s.index('(') + 1:s.index(',List(')] for s in data[2:5]]
                text = data[5]
                curExtraction = Extraction(pred = rel, head_pred_index = -1, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d



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


    oie = OpenieFourReader()
    oie.read(inp_fn)
    oie.output_tabbed(out_fn)

    logging.info("DONE")
