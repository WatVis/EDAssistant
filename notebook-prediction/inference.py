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

class RetrivalDB:
  def __init__(self):
    self.embed = np.load("codebase_embed.npy", allow_pickle=True)
    self.kernel_ids = np.load("codebase_id.npy", allow_pickle=True)

  def getDoc(self, raw_idx):
    if raw_idx < 0 or raw_idx >= self.embed.shape[0]:
      print("ERROR: out of index")
      return None
    kid = self.kernel_ids[raw_idx]
    path = kid.split('##')[0]
    lineno = int(kid.split('##')[1])
    return path, lineno

  def find_sim(self, embed, topn=10):
    result = np.einsum("ij,ij->i",self.embed,embed)
    rank = np.argsort(-result)
    doc_list = [self.getDoc(r) + (result[r],) for r in rank]
    return doc_list

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
  tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
  model = RobertaModel.from_pretrained(model_name_or_path)    
  model=BertModel(model).to(device)
  checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
  output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
  model.load_state_dict(torch.load(output_dir),strict=False)  

  # gen = torch.load("./gen_consine/best_gen.pt").to(device)
  gen = Generator(768, 768).to(device)
  gen.load_state_dict(torch.load('./gen_saved/best_gen_state_dict.pt'))
  gen.eval()
  model.eval()
  db = RetrivalDB()
  
  with torch.no_grad():
    while(True):
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      input("Update the sample.py and press Enter to continue...")
      # TODO: reads ipynb
      input_file = './sample.ipynb'
      embed_list = [torch.zeros((1,768)).to(device)]
      f = codecs.open(input_file, 'r')
      source = f.read()

      y = json.loads(source)
      for x in y['cells']:
          code = ""
          for x2 in x['source']:
              if x2[-1] != '\n':
                  x2 = x2 + '\n'
              code += x2
          embed_list.append(get_embedding(code, device, model))

      # with open(input_file, encoding='utf-8') as f:
      #   # treat as one cell for now
      #   embed_list.append(get_embedding(''.join(f.readlines()), device, model))

      predict_embed = gen.generate_embedding(embed_list)

      print("??????", len(embed_list), predict_embed.mean())

      predict_embed = [embed.detach().cpu().numpy() for embed in predict_embed]
      
      doc_list = db.find_sim(predict_embed)

      file_path = '../../kaggle-dataset/sliced-notebooks-full-new'

      df_list = []
      for kernel_id, cell_no, sim in doc_list:
        competiton = kernel_id.split('\\')[0]
        kid = kernel_id.split('\\')[1].split('_')[0]
        subid = kernel_id.split('\\')[1].split('_')[1]
        df_list.append((competiton, kid, subid,cell_no, sim))
      df = pd.DataFrame(df_list, columns=["competiton", "kid", "subid","cell_no", "sim"])
      # df = df.drop_duplicates(subset=["kid"])
      count = 0
      file_path = '../../kaggle-dataset/sliced-notebooks-full-new'
      print(df.head(100))
      check_df = df.loc[df["competiton"]==("bengaliai-cv19")]
      print(check_df.head(5))
      print(check_df.loc[check_df["kid"] == "25847047"])
      for _, row in df.iterrows():
        count += 1
        if count > 10:
          break

        full_id = '{}/{}_{}'.format(row["competiton"], row["kid"], row["subid"])
        source_path = '{}/{}.py'.format(file_path, full_id)
        meta_path = '{}/{}.csv'.format(file_path, full_id)

        cell_no = row["cell_no"]

        print("***KERNEL:", row["kid"])
        print("***PATH:", source_path, meta_path)
        print("***cell_no", row["cell_no"])
        print("***SIM", row["sim"])
        meta_df = pd.read_csv(meta_path)
        cell_list = []
        for _, meta_row in meta_df.iterrows():
          cell_list.append((meta_row['CELL'], meta_row['USAGE']))
        print("***FUNCTIONS:", cell_list[cell_no][1])
        with open(source_path, encoding='utf-8') as f:
          start = 0
          if cell_no != 0:
            start = cell_list[cell_no-1][0]
          end = cell_list[cell_no][0]
          print("lineno start at:", start)
          print(''.join(f.readlines()[start:end]))

      # pickle.dump(doc_list,open("doc_list.pkl",'wb'))

      # for kernel_id, cell_no, sim in doc_list:
      #   print("############################")
      #   kernel_id = '/'.join(kernel_id.split('\\'))
      #   source_path = '{}/{}.py'.format(file_path, kernel_id)
      #   meta_path = '{}/{}.csv'.format(file_path, kernel_id)
      #   print("***KERNEL:", kernel_id)
      #   print("***PATH:", source_path, meta_path)
      #   print("***cell_no", cell_no)
      #   print("***SIM", sim)
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

    
        



