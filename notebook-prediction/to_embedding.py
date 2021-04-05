import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import BertModel
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

from utils import CellFeatures, InputFeatures
from config import *

from tqdm import tqdm, trange      

def to_embedding(data, model):
  code_inputs = data[0].to(device)
  attn_mask = data[1].to(device)
  position_idx = data[2].to(device)  
  with torch.no_grad():
    code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
  return code_vec

class TextDataset(Dataset):
    def __init__(self, cache_file, model):
        self.raw=pickle.load(open(cache_file,'rb'))    
        self.set_max_size(-1)
        self.max_size = -1
        self.get_embedding = True
        self.model = model
                
    def __len__(self):
        return len(self.examples)

    def set_max_size(self, size):
      self.max_size = size
      if size == -1:
        self.examples = self.raw
      else:
        self.examples=[ex for ex in self.raw if ex.size() <= size] 

    def __getFromFeatures(self, item):
      #calculate graph-guided masked function
      attn_mask=np.zeros((code_length+data_flow_length,
                          code_length+data_flow_length),dtype=bool)
      #calculate begin index of node and max length of input
      node_index=sum([i>1 for i in item.position_idx])
      max_length=sum([i!=1 for i in item.position_idx])
      #sequence can attend to sequence
      attn_mask[:node_index,:node_index]=True
      #special tokens attend to all tokens
      for idx,i in enumerate(item.code_ids):
          if i in [0,2]:
              attn_mask[idx,:max_length]=True
      #nodes attend to code tokens that are identified from
      for idx,(a,b) in enumerate(item.dfg_to_code):
          if a<node_index and b<node_index:
              attn_mask[idx+node_index,a:b]=True
              attn_mask[a:b,idx+node_index]=True
      #nodes attend to adjacent nodes 
      for idx,nodes in enumerate(item.dfg_to_dfg):
          for a in nodes:
              if a+node_index<len(item.position_idx):
                  attn_mask[idx+node_index,a+node_index]=True  
                  
      return (torch.tensor(item.code_ids).view(1, -1),
            torch.tensor(attn_mask).view(1, 320, 320),
            torch.tensor(item.position_idx).view(1, -1))

    def __getitem__(self, idx): 
      input_feautres = self.examples[idx].input_features
      kernel_id = self.examples[idx].kernel_id
      lib_names = []
      code_inputs = [] 
      attn_mask = []
      position_idx = []  
      for item in input_feautres:
        lib_names.append(item.libs_names)
        c, a, p = self.__getFromFeatures(item)
        code_inputs.append(c)
        attn_mask.append(a)
        position_idx.append(p)

      code_inputs = torch.cat(code_inputs, dim=0)
      attn_mask = torch.cat(attn_mask, dim=0)
      position_idx = torch.cat(position_idx, dim=0)

      lib_names = np.array(lib_names, dtype=object)

      if self.get_embedding:
        return to_embedding((code_inputs, attn_mask, position_idx), self.model), kernel_id, lib_names
      else:
        return (code_inputs, attn_mask, position_idx), kernel_id, lib_names

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
  model = RobertaModel.from_pretrained(model_name_or_path)    
  model = BertModel(model).to(device)
  checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
  output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
  model.load_state_dict(torch.load(output_dir),strict=False)    
  print("loading dataset...")
  dataset = TextDataset('./test.pkl', model)
  print("start computing...")
  # seq_length = 12
  dataset.set_max_size(12)
  all_tensors = []
  all_kernel_ids = []
  all_libnames = []
  for idx in trange(len(dataset)):
    d = dataset[idx]
    all_tensors.append(d[0].cpu().detach().numpy())
    all_kernel_ids.append(d[1])
    all_libnames.append(np.array(d[2]))
  np.save("./embed_tensors_test", np.array(all_tensors, dtype=object))
  np.save("./kernel_ids_test", np.array(all_kernel_ids, dtype=object))
  np.save("./lib_names_test", np.array(all_libnames, dtype=object))
