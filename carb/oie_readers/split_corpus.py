""" Usage:
    split_corpus --corpus=CORPUS_FN --reader=READER --in=INPUT_FN --out=OUTPUT_FN

Split OIE extractions according to raw sentences.
This is used in order to split a large file into train, dev and test.

READER - points out which oie reader to use (see dictionary for possible entries)
"""
from clausieReader import ClausieReader
from ollieReader import OllieReader
from openieFourReader import OpenieFourReader
from propsReader import PropSReader
from reVerbReader import ReVerbReader
from stanfordReader import StanfordReader
from docopt import docopt
import logging
logging.basicConfig(level = logging.INFO)

available_readers = {
    "clausie": ClausieReader,
    "ollie": OllieReader,
    "openie4": OpenieFourReader,
    "props": PropSReader,
    "reverb": ReVerbReader,
    "stanford": StanfordReader
}


if __name__ == "__main__":
    args = docopt(__doc__)
    inp = args["--in"]
    out = args["--out"]
    corpus = args["--corpus"]
    reader = available_readers[args["--reader"]]()
    reader.read(inp)
    reader.split_to_corpus(corpus,
                           out)
