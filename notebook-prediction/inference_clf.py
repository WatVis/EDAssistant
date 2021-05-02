import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import pandas as pd
from model import BertModel, Generator, LibClassifier
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

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
  tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
  model = RobertaModel.from_pretrained(model_name_or_path)    
  model=BertModel(model).to(device)
  checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
  output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
  model.load_state_dict(torch.load(output_dir),strict=False)  

  lib_dict = pickle.load(open("lib_dict.pkl",'rb'))   
  lib_dict = {v: k for k, v in lib_dict.items()}
  # clf = torch.load("./clf_jaccard/best_clf.pt").to(device)
  gen = Generator(768, 768).to(device)
  clf = LibClassifier(gen, 768, 16855).to(device)
  clf.load_state_dict(torch.load('./clf_jaccard/best_clf_state_dict.pt'))
  clf.eval()
  # model.eval()
  
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
          for x2 in x['source']:
              if x2[-1] != '\n':
                  x2 = x2 + '\n'
              embed_list.append(get_embedding(x2, device, model))

      predict_embed = clf.classify(embed_list)
      values, idxs = torch.topk(predict_embed, 5)
      idxs = idxs.detach().cpu().numpy()[0]
      print(idxs, values)
      print([lib_dict[i] for i in idxs])

      
      
      

    
        



