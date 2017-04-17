import random, sys, os
from rnnslu.load import atisfold

class ATISReader () :
    """Tools to read the ATIS pickles and dump to text.
    """
    def __init__ (self) :
        pass

    def _WriteFile (self, data, folder, mapper, fname) :
        opath = os.path.join (folder, fname)
        ofp = open (opath, "w")
        for item in data :
            sentence = u" ".join ([mapper [w] for w in item])
            ofp.write ("{0}\n".format (sentence.encode ("utf8")))
        ofp.close ()
        print >> sys.stderr, opath

        return

    def ProcessTrainingData (self, conf) :
        folder = conf ["indir"]
        if not os.path.exists (folder) :
            os.mkdir (folder)

        # load the dataset
        train_set, valid_set, test_set, dic = atisfold (conf['fold'])
        idx2label = dict ((k,v) for v,k in dic ['labels2idx'].iteritems ())
        idx2word  = dict ((k,v) for v,k in dic ['words2idx'].iteritems ())
        idx2table = dict ((k,v) for v,k in dic ['tables2idx'].iteritems ())

        train_lex, train_ne, train_y = train_set
        valid_lex, valid_ne, valid_y = valid_set
        test_lex,  test_ne,  test_y  = test_set

        self._WriteFile (train_lex, folder, idx2word, "train_lex")
        self._WriteFile (train_ne, folder, idx2table, "train_ne")
        self._WriteFile (train_y, folder, idx2label, "train_y")

        self._WriteFile (valid_lex, folder, idx2word, "valid_lex")
        self._WriteFile (valid_ne, folder, idx2table, "valid_ne")
        self._WriteFile (valid_y, folder, idx2label, "valid_y")

        self._WriteFile (test_lex, folder, idx2word, "test_lex")
        self._WriteFile (test_ne, folder, idx2table, "test_ne")
        self._WriteFile (test_y, folder, idx2label, "test_y")

        return

if __name__ == "__main__" :
    import sys, os

    config = {
        'fold': 3, # 5 folds 0,1,2,3,4
        'lr': 0.0627142536696559,
        'verbose': 1,
        'decay': False, # decay on the learning rate if improvement stops
        'win': 5, # 7 number of words in the context window
        'bs': 7, # 9 number of backprop through time steps
        'nhidden': 50, # number of hidden units
        'seed': 345,
        'emb_dimension': 50, # dimension of word embedding
        'nepochs': 50,
        'indir': "train_slu_new"
        }

    atis = ATISReader ()
    atis.ProcessTrainingData (config)
