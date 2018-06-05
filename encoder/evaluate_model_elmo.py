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
import numpy as np
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

from xutils import dotdict

from ipdb import set_trace

PATH_TO_DATA = "/data/work/jingbo/ll2/SentEval/data"

parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--encoder_path", type=str, default='../savedir/SNLI/elmo-lstm-max.pickle.encoder',
                    help="path to elmo-lstm-max")
parser.add_argument("--elmo_path", type=str, default='../savedir/SNLI/elmo-lstm-max.pickle.scalar',
                    help="path to elmo-lstm-max")
parser.add_argument("--gpu_id", type=int, default=5, help="GPU ID")
params, _ = parser.parse_known_args()


torch.cuda.set_device(params.gpu_id)

# import senteval
# sys.path.insert(0, PATH_SENTEVAL)
import senteval

def get_elmo_rep(batch, elmo, if_cuda=True):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    batch = np.array(batch)

    char_ids = batch_to_ids(batch)
    if if_cuda:
        char_ids = char_ids.cuda()
    embeddings = elmo(char_ids)['elmo_representations'][0].transpose(0, 1)

    return embeddings, lengths

def prepare(params, samples, name):

    params.elmo_model.cuda()
    params.nli_net.cuda()

    params.elmo_model.eval()
    params.nli_net.eval()
    return

def batcher(params, batch, name):
    # batch contains list of words
    batch_elmo, batch_length = get_elmo_rep(batch, params.elmo_model)
    embeddings = params.nli_net((batch_elmo, batch_length))

    # embeddings = torch.unbind(embeddings.data, 0)

    return embeddings.data.cpu().numpy()


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define transfer tasks
transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16','CR',
                    'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC','MR',
                    'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']

# define senteval params

params_senteval = dotdict({'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}})

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load model

    params_senteval.nli_net = torch.load(params.encoder_path)
    params_senteval.elmo_model = torch.load(params.elmo_path)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    print(results_transfer)
