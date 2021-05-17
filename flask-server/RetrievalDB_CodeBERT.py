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
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from pathlib import Path
import codecs
import json

from utils import CellFeatures, InputFeatures, convert_examples_to_features, parseNotebook, get_notebook_list, get_embedding
from config import *

class RetrievalDB_CodeBERT:
  def __init__(self, filepath = ''):
    self.filepath = filepath
    self.embed = np.load(filepath+"embed_tensors.npy", allow_pickle=True)
    self.kernel_ids = np.load(filepath+"kernel_ids.npy", allow_pickle=True)
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
      print("ERROR: out of index")
      return None
    first = 0
    last = len(self.idx_list) - 1
    midpoint = (first + last)//2 
    while True:
      midpoint = (first + last)//2            
      if self.idx_list[midpoint] <= raw_idx and self.idx_list[midpoint+1] > raw_idx:
        break
      else:
        if raw_idx < self.idx_list[midpoint]:
          last = midpoint-1
        else:
          first = midpoint+1  
    kernel_id = self.kernel_ids[midpoint]
    return (kernel_id, raw_idx - self.idx_list[midpoint])

  def find_sim(self, embed, topn=10):
    result = np.einsum("ij,ij->i",self.raw,embed)
    rank = np.argsort(-result)[:topn]
    doc_list = [self.getDoc(r) for r in rank]
    return doc_list

def inferenceRNN_CodeBERT(notebookSrc, retrievalDB_cb, filepath=''):
  
    # Require torch.no_grad so that excessive GPU memory is not used 
    with torch.no_grad():

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
      tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
      model = RobertaModel.from_pretrained(model_name_or_path)    
      model=BertModel(model).to(device)
      checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
      output_dir = os.path.join(filepath + 'saved_models/python', '{}'.format(checkpoint_prefix))  
      model.load_state_dict(torch.load(output_dir),strict=False)  

      gen = torch.load(filepath + "gen_consine/best_gen.pt").to(device)
      gen.eval()
      model.eval()
      db = retrievalDB_cb

      embed_list = []
      y = notebookSrc
      for x in y['cells']:
          for x2 in x['source']:
              if x2[-1] != '\n':
                  x2 = x2 + '\n'
              embed_list.append(get_embedding(x2, device, model))
      # with open(input_file, encoding='utf-8') as f:
      #   # treat as one cell for now
      #   embed_list.append(get_embedding(''.join(f.readlines()), device, model))

      predict_embed = gen.generate_embedding(embed_list)

      predict_embed = [embed.detach().cpu().numpy() for embed in predict_embed]

      doc_list = db.find_sim(predict_embed)

      return doc_list

      # file_path = '../kaggle-dataset/sliced-notebooks-full-new'

      # for kernel_id, cell_no in doc_list:
      #   print("############################")
      #   kernel_id = '/'.join(kernel_id.split('\\'))
      #   source_path = '../kaggle-dataset/sliced-notebooks-full-new/{}.py'.format(kernel_id)
      #   meta_path = '../kaggle-dataset/sliced-notebooks-full-new/{}.csv'.format(kernel_id)
      #   print("***KERNEL:", kernel_id)
      #   print("***PATH:", source_path, meta_path)
      #   print("***cell_no", cell_no)
      #   df = pd.read_csv(meta_path)
      #   cell_list = []
      #   for index, row in df.iterrows():
      #     cell_list.append((row['CELL'], row['USAGE']))

      #   print("***FUNCTIONS:", cell_list[cell_no][1])
      #   with open(source_path, encoding='utf-8') as f:
      #     start = 0
      #     if cell_no != 0:
      #       start = cell_list[cell_no-1][0]
      #     end = cell_list[cell_no][0]
      #     print("lineno start at:", start)
      #     # if cell_no == len(cell_list) - 1:
      #     #   print(''.join(f.readlines()[start:]))
      #     # else:
      #     print(''.join(f.readlines()[start:end]))


          



