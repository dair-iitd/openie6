import nltk
import sys

content = open(sys.argv[1], 'r').read()
open(sys.argv[2],'w').write('\n'.join(nltk.tokenize.sent_tokenize(content)))

