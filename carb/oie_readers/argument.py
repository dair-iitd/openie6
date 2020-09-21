import nltk
from operator import itemgetter

class Argument:
    def __init__(self, arg):
        self.words = [x for x in arg[0].strip().split(' ') if x]
        self.posTags = map(itemgetter(1), nltk.pos_tag(self.words))
        self.indices = arg[1]
        self.feats = {}

    def __str__(self):
        return "({})".format('\t'.join(map(str,
                                           [escape_special_chars(' '.join(self.words)),
                                            str(self.indices)])))

COREF = 'coref'

## Helper functions
def escape_special_chars(s):
    return s.replace('\t', '\\t')

