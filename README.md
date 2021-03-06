Refactoring
===========

This is a refactoring of the original mesnilgr/is13 repository.  The conlleval.pl script has been ported
to pure python, and the base refactored for arguably easier applicability to a new data. Also ports most major
functionality of conlleval.pl to native python.

Idea is to make it a bit more straight-forward to take new training data and map it to the input text-format
files, rather than the ATIS pickles:
```
$ sudo python setup.py install
$ get_atis_data -i train_slu_data
$ train_slu -i train_slu_data
NEW BEST: epoch 0 valid F1 85.4118729867 best test F1 81.1594202899
...
```

Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding
==============================================================================================================

### Code for RNN and Spoken Language Understanding

Based on the Interspeech '13 paper:

[Grégoire Mesnil, Xiaodong He, Li Deng and Yoshua Bengio - **Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/RNNSpokenLanguage2013.pdf)

We also have a follow-up IEEE paper:

[Grégoire Mesnil, Yann Dauphin, Kaisheng Yao, Yoshua Bengio, Li Deng, Dilek Hakkani-Tur, Xiaodong He, Larry Heck, Gokhan Tur, Dong Yu and Geoffrey Zweig - **Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU_final_doubleColumn.pdf)

## Code

This code allows to get state-of-the-art results and a significant improvement
(+1% in F1-score) with respect to the results presented in the paper.

In order to reproduce the results, make sure Theano is installed and the
repository is in your `PYTHONPATH`, e.g run the command
`export PYTHONPATH=/path/where/is13/is:$PYTHONPATH`. Then, run the following
commands:

```
git clone git@github.com:mesnilgr/is13.git
python is13/examples/elman-forward.py
```

For running the Jordan architecture:

```
python is13/examples/jordan-forward.py
```

## ATIS Data

[Download ATIS Dataset here!](https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0) [split 0](https://www.dropbox.com/s/4n6wkptdazthj0e/atis.fold0.pkl?dl=0) [split 1](https://www.dropbox.com/s/ikyzbb06mxh69pv/atis.fold1.pkl?dl=0) [split 2](https://www.dropbox.com/s/5ek0vx2yo54878o/atis.fold2.pkl?dl=0) [split 3](https://www.dropbox.com/s/31r4bqids24n0vr/atis.fold3.pkl?dl=0) [split 4](https://www.dropbox.com/s/hvp50m4snmkroe4/atis.fold4.pkl?dl=0)

```
import cPickle
train, test, dicts = cPickle.load(open("atis.pkl"))
```

`dicts` is a python dictionnary that contains the mapping from the labels, the
name entities (if existing) and the words to indexes used in `train` and `test`
lists. Refer to this [tutorial](http://deeplearning.net/tutorial/rnnslu.html) for more details. 

Running the following command can give you an idea of how the data has been preprocessed:

```
python data/load.py
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Recurrent Neural Network Architectures for Spoken Language Understanding</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Grégoire Mesnil</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/mesnilgr/is13" rel="dct:source">https://github.com/mesnilgr/is13</a>.
