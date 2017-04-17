#!/usr/bin/env python
import numpy, time, sys, os, random, re
from jordan import model
from CoNLLeval import CoNLLeval


class SLUNetTrainer () :
    def __init__ (self, lr=0.0627142536696559, verbose=1, decay=False,
                  window=5, batch_size=7, nhidden=50, seed=345, emb_dimension=50,
                  nepochs=50, indir="train_slu", fold=3) :
        self.lr            = lr
        self.clr           = lr
        self.verbose       = verbose
        self.decay         = decay
        self.window        = window
        self.batch_size    = batch_size
        self.nhidden       = nhidden
        self.seed          = seed
        self.nepochs       = nepochs
        self.indir         = indir
        self.fold          = fold
        self.emb_dimension = emb_dimension
        
        numpy.random.seed (self.seed)
        random.seed (self.seed)

        self.data = self.LoadTrainingData ()
        self.rnn  = model (
            nh = self.nhidden,
            nc = self.data ["nclasses"],
            ne = self.data ["vocsize"],
            de = self.emb_dimension,
            cs = self.window
            )

    def _Shuffle (self) :
        """Shuffle each list in a list of lists.

        Shuffle each list in a list of lists, using a user-provided
        seed to ensure that all lists are shuffled in the same order.
        In addition, each list is shuffled in place.

        Args:
            list_of_lists (list): The list of lists to shuffle.
            seed (int): The seed to employ with the random number generator.

        """        
        for ilist in ["train_lex", "train_ne", "train_y"] :
            random.seed (self.seed)
            random.shuffle (self.data [ilist])
        
    def _LoadFile (self, fname, mapper, inv_mapper) :
        """Load a training file and extend the idx/token maps.

        Load a traiing file and extend the specified idx/token
        maps for the global data models.

        Args:
            fname (str): Input data file.
            mapper (dict): Map from tokens to ids
            inv_mapper (dict): Map from ids to tokens

        Returns:
            dict: Map from ids to tokens
            dict: Map from tokens to ids
            list: Training data elements.
        
        """
        data_path = os.path.join (self.indir, fname)
        data_ifp  = open (data_path, "r")
        data = []
        with open (data_path, "r") as ifp :
            for line in ifp :
                words = re.split (ur"\s+", line.decode ("utf8").strip ())
                for word in words :
                    idx = len (mapper)
                    if not word in mapper :
                        mapper [word] = idx
                data.append (numpy.array ([mapper [w] for w in words]))
            
        inv_mapper.update (dict ((key,val) for val,key in mapper.iteritems ()))
        return inv_mapper, mapper, data

    def LoadTrainingData (self) :
        """Load all component training data files.

        Load all component training data files.
        
        """
        if not os.path.exists (self.indir) : 
            os.mkdir (self.indir)
            
        data = {
            #Training data
            "train_lex": [], "train_ne": [], "train_y": [],
            #Validation data
            "valid_lex": [], "valid_ne": [], "valid_y": [], 
            #Test data
            "test_lex": [], "test_ne": [], "test_y": [], 
            #Mappers
            "word2idx":  {}, "idx2word": {}, 
            "label2idx": {}, "idx2label": {}, 
            "table2idx": {}, "idx2table": {},
            #Counters
            "vocsize":    0,
            "nclasses":   0,
            "nsentences": 0,
            }

        for fname in ["train_lex", "valid_lex", "test_lex"] :
            data ["idx2word"], data ["word2idx"], items \
                = self._LoadFile (fname, data ["word2idx"], data ["idx2word"])
            data [fname].extend (items)

        for fname in ["train_ne", "valid_ne", "test_ne"] :
            data ["idx2table"], data ["table2idx"], items \
                = self._LoadFile (fname, data ["table2idx"], data ["idx2table"])
            data [fname].extend (items)

        for fname in ["train_y", "valid_y", "test_y"] : 
            data ["idx2label"], data ["label2idx"], items \
                = self._LoadFile (fname, data ["label2idx"], data ["idx2label"])
            data [fname].extend (items)

        data ["vocsize"]    = len (data ["word2idx"])
        data ["nclasses"]   = len (data ["label2idx"])
        data ["nsentences"] = len (data ["train_lex"])
        
        return data

    def Predict (self, input) :
        """Predict the output slots for a single input vector.

        Predict the output slots for a single input vector.

        Args:
            input (list): The input sentence as a list of idx reps.

        Returns:
            list: The predicted set of labels for the sentence.

        """
        prediction = map (
            lambda idx: self.data ["idx2label"] [idx], 
            self.rnn.classify (
                numpy.asarray (
                    self._ContextWindow (input)
                    ).astype ('int32')
                )
            )
            
        return prediction

    def PredictBatch (self, inputs) :
        """Apply ```Predict``` to a batch of sentences in sequence.

        Apply ```Predict``` to a batch of sentences in sequence.

        Args:
            inputs (list): A list of lists.

        Returns:
            list: The list of list of prediction results.
            
        """
        results = []
        for input in inputs :
            results.append (self.Predict (input))
            
        return results

    def _ContextWindow (self, ilist):
        '''
        win :: int corresponding to the size of the window
        given a list of indexes composing a sentence
        it will return a list of list of indexes corresponding
        to context windows surrounding each word in the sentence
        '''
        assert (self.window % 2) == 1
        assert self.window >= 1
        ilist = list (ilist)

        lpadded = self.window/2 * [-1] + ilist + self.window/2 * [-1]
        out = [lpadded [i:i + self.window] for i in range (len (ilist))]

        assert len (out) == len (ilist)

        return out

    def _Minibatch (self, ilist) :
        """Return a list of minibatches of indexes.

        Return a list of minibatches of indexes where the size
        of each is equal to ```batch_size```.  Border cases are
        treated as follows:
            Example: in> [0,1,2,3] and batch_size = 3
                out> [0],[0,1],[0,1,2],[1,2,3]]

        Args:
            ilist (list): The list of indexes to minibatch-ify
            batch_size (int): The maximum minibatch size.

        Returns
            list: The minibatch-ified list-of-lists

        """
        out  = [ilist [:i] for i in xrange (1, min (self.batch_size, len (ilist) + 1))]
        out += [ilist [i - self.batch_size:i] for i in xrange (self.batch_size, len(ilist) + 1)]

        assert len (ilist) == len (out)

        return out

    def _Evaluate (self, predictions, groundtruth, words, ofile) :
        ce = CoNLLeval ()
        opath   = os.path.join (self.indir, ofile)
        results = ce.conlleval (predictions, groundtruth, words, opath)

        return results

    def EvaluateProgress (self, predictions_test, groundtruth_test, words_test,
                          predictions_valid, groundtruth_valid, words_valid, 
                          epoch) :
        res_valid = self._Evaluate (
            predictions_valid,
            groundtruth_valid,
            words_valid,
            "current.valid.txt"
            )
        
        res_test  = self._Evaluate (
            predictions_test,
            groundtruth_test,
            words_test,
            "current.test.txt"
            )

        if res_valid ["FB1"] > self.best_f1 :
            self.rnn.save (self.indir)
            self.best_f1 = res_valid ["FB1"]
            if self.verbose > 0 :
                print 'NEW BEST: epoch', epoch, 'valid F1', \
                    res_valid["FB1"], 'best test F1', \
                    res_test["FB1"], ' '*20
            
            self.vf1 = res_valid ["FB1"]
            self.vp  = res_valid ["precision"]
            self.vr  = res_valid ['recall']

            self.tf1 = res_test ["FB1"]
            self.tp  = res_test ["precision"]
            self.tr  = res_test ["recall"]

            self.be = epoch

            os.rename (
                os.path.join (self.indir, 'current.test.txt'), 
                os.path.join (self.indir, 'best.test.txt')
                )
            os.rename (
                os.path.join (self.indir, 'current.valid.txt'), 
                os.path.join (self.indir, 'best.valid.txt')
                )

        return 

    def TrainSLUNet (self) :    
        # train with early stopping on validation set
        self.best_f1 = -numpy.inf
    
        for epoch in xrange (self.nepochs) :
            self._Shuffle ()
            self.ce = epoch
            epoch_start = time.time ()
            for index in xrange (self.data ["nsentences"]) :
                cwords = self._ContextWindow (self.data ["train_lex"][index])
                words  = map (
                    lambda x: numpy.asarray (x).astype ('int32'),
                    self._Minibatch (cwords)
                    )
                labels = self.data ["train_y"][index]

                for word_batch, label_last_word in zip (words, labels) :
                    self.rnn.train (word_batch, label_last_word, self.clr)
                    self.rnn.normalize ()
                
                if self.verbose > 0 :
                    update_str = "[learning] Epoch {0} >> {1:.2f} "\
                        "Completed in {2:.2f}s << \r"
                    update_str = update_str.format (
                        epoch, 
                        (index + 1) * 100. / self.data ["nsentences"],
                        time.time () - epoch_start
                        )
                    print update_str,
                    sys.stdout.flush ()
            
            # evaluation // back into the real world : idx -> words
            predictions_test = self.PredictBatch (self.data ["test_lex"])
            groundtruth_test = [
                map (lambda x: self.data ["idx2label"][x], y) \
                    for y in self.data ["test_y"]
                ]
            words_test = [
                map (lambda x: self.data ["idx2word"][x], w) \
                    for w in self.data ["test_lex"]
                ]

            predictions_valid = self.PredictBatch (self.data ["valid_lex"])
            groundtruth_valid = [
                map (lambda x: self.data ["idx2label"][x], y) \
                    for y in self.data ["valid_y"]
                ]
            words_valid = [
                map (lambda x: self.data ["idx2word"][x], w) \
                    for w in self.data ["valid_lex"]
                ]

            #Evaluate progress // compute the accuracy using CoNLLeval
            self.EvaluateProgress (
                predictions_test, groundtruth_test, words_test, 
                predictions_valid, groundtruth_valid, words_valid,
                epoch
                )
        
            # learning rate decay if no improvement in 10 epochs
            if self.decay and abs (self.be - self.ce) >= 10 : self.clr *= 0.5 
            if self.clr < 1e-5: break

        print 'BEST RESULT: epoch', epoch, 'valid F1', self.vf1, \
            'best test F1', self.tf1, 'with the model', self.indir

if __name__ == '__main__':


    trainer = SLUNetTrainer ()
    trainer.TrainSLUNet ()
