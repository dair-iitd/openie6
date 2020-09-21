class OieReader:

    def read(self, fn, includeNominal):
        ''' should set oie as a class member
        as a dictionary of extractions by sentence'''
        raise Exception("Don't run me")

    def count(self):
        ''' number of extractions '''
        return sum([len(extractions) for _, extractions in self.oie.items()])

    def split_to_corpus(self, corpus_fn, out_fn):
        """
        Given a corpus file name, containing a list of sentences
        print only the extractions pertaining to it to out_fn in a tab separated format:
        sent, prob, pred, arg1, arg2, ...
        """
        raw_sents = [line.strip() for line in open(corpus_fn)]
        with open(out_fn, 'w') as fout:
            for line in self.get_tabbed().split('\n'):
                data = line.split('\t')
                sent = data[0]
                if sent in raw_sents:
                    fout.write(line + '\n')

    def output_tabbed(self, out_fn):
        """
        Write a tabbed represenation of this corpus.
        """
        with open(out_fn, 'w') as fout:
            fout.write(self.get_tabbed())

    def get_tabbed(self):
        """
        Get a tabbed format representation of this corpus (assumes that input was
        already read).
        """
        return "\n".join(['\t'.join(map(str,
                                        [ex.sent,
                                         ex.confidence,
                                         ex.pred,
                                         '\t'.join(ex.args)]))
                          for (sent, exs) in self.oie.iteritems()
                          for ex in exs])

