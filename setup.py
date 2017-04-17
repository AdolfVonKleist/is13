#!/usr/bin/python
from setuptools import setup

#Install rnnslu
setup (
    name         = 'rnnslu',
    version      = '0.1',
    description  = 'RNN SLU python package based on mesnilgr/is13.',
    url          = 'https://github.com/AdolfVonKleist/rnn-slu',
    author       = 'Josef Novak',
    author_email = 'josef.robert.novak@gmail.com',
    license      = 'CC BY-NC 4.0',
    packages     = ['rnnslu'],
    install_requires=[
        "theano",
    ],
    scripts=[
        'bin/train_slu',
        'bin/get_atis_data'
    ],
    package_data = {},
    zip_safe     = False,
    include_package_data = True,
)
