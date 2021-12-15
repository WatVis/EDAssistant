#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import pandas as pd
from model import BertModel, Generator
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, \
    RandomSampler, TensorDataset
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from pathlib import Path
import codecs
import json

from utils import CellFeatures, InputFeatures, \
    convert_examples_to_features, parseNotebook, get_notebook_list, \
    get_embedding
from config import *

from tokenize_code import tokenize_code
from gensim.models.doc2vec import Doc2Vec




class RetrievalDB_doc2vec:

    def __init__(self):
        self.embed = np.load('./bigfiles/embed_tensors_clean_apr29.npy',
                             allow_pickle=True)
        self.kernel_ids = np.load('bigfiles/baseline_rnn_clf/train_clf_data/kernel_ids_apr29.npy',
                                  allow_pickle=True)
        # Load the doc2vec model
        self.model = Doc2Vec.load("./bigfiles/doc2vec_model/notebook-doc2vec-model-apr24.model")

        
        self.idx_list = []
        idx = 0
        doc_list = []
        for doc in self.embed:
            self.idx_list.append(idx)
            idx += doc.shape[0]
            doc_list.append(doc)
        self.raw = np.concatenate(doc_list)

    def getDoc(self, raw_idx):
        if raw_idx < 0 or raw_idx >= self.raw.shape[0]:
            print('ERROR: out of index')
            return None
        first = 0
        last = len(self.idx_list) - 1
        midpoint = (first + last) // 2
        while True:
            midpoint = (first + last) // 2
            if self.idx_list[midpoint] <= raw_idx \
                and self.idx_list[midpoint + 1] > raw_idx:
                break
            else:
                if raw_idx < self.idx_list[midpoint]:
                    last = midpoint - 1
                else:
                    first = midpoint + 1
        kernel_id = self.kernel_ids[midpoint]
        return (kernel_id, int(raw_idx - self.idx_list[midpoint])) # need to cast index from int64 to int in order to jsonify

    def find_sim(self, embed, topn=10):
        result = np.einsum('ij,ij->i', self.raw, embed)
        rank = np.argsort(-result)[:topn]
        doc_list = [self.getDoc(r) for r in rank]
        return doc_list

def inferenceRNN_doc2vec(notebookSrc, retrievalDB):
    '''
    Infer the next code cell of a notebook with file specified by filepath
    Arguments: filepath : The path to the file that inference should be performed on. 
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the RNN model
    gen = torch.load('./bigfiles/baseline_rnn_clf/gen_saved_rnn/best_gen.pt').to(device)
    
    db = retrievalDB
    
    embed_list = []
    y = notebookSrc
    for x in y['cells']:
        for x2 in x['source']:
            if x2[-1] != '\n':
                x2 = x2 + '\n'
            embed_list.append(torch.Tensor(db.model.infer_vector(tokenize_code(x2,'code'))).to(device))
    predict_embed = gen.generate_embedding(embed_list)
    predict_embed = [embed.detach().cpu().numpy() for embed in predict_embed]
    
    doc_list = db.find_sim(predict_embed, topn=10)
    return doc_list    