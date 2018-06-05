# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import argparse
import logging

import torch

from xutils import dotdict


GLOVE_PATH = "../dataset/GloVe/glove.840B.300d.txt"
PATH_TO_DATA = "/data/work/jingbo/ll2/SentEval/data"

# assert os.path.isfile(GLOVE_PATH) and PATH_SENTEVAL and PATH_TRANSFER_TASKS, 'Set PATHs'
assert os.path.isfile(GLOVE_PATH) and PATH_TO_DATA, 'Set PATHs'

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--modelpath", type=str, default='../savedir/blstm-max.pickle.encoder',
                    help="path to model")
params, _ = parser.parse_known_args()


# import senteval
# sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples, name):
    params.infersent.build_vocab([' '.join(s) for s in samples],
                                 tokenize=False)


def batcher(params, batch, name):
    # batch contains list of words
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size,
                                         tokenize=False)
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16','CR',
                    'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC','MR',
                    'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']

# define senteval params

params_senteval = dotdict({'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10, 'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load model
    params_senteval.infersent = torch.load(params.modelpath)
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
