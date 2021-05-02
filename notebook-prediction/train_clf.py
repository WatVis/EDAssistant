import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Generator, LibClassifier
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from tqdm import tqdm, trange
import multiprocessing
cpu_cont = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class cacheDataset(Dataset):
  def __init__(self):
    self.embed = np.load("embed_tensors.npy", allow_pickle=True)
    self.kernel_ids = np.load("kernel_ids.npy", allow_pickle=True)
    self.lib_names = np.load("lib_names.npy", allow_pickle=True)
    self.lib_dict = pickle.load(open("lib_dict.pkl",'rb'))    

  def __len__(self):
    return len(self.embed)

  def getMultiLabel(self, lib_list):
    label = np.zeros(16855)
    for lib in lib_list:
      label[self.lib_dict[lib]] = 1
    return label.reshape((1, 16855))

  def __getitem__(self, idx):
    embed =  np.vstack([np.zeros((1,768)), self.embed[idx]])
    lib_name = self.lib_names[idx]
    lib_name = np.concatenate([self.getMultiLabel(lib) for lib in lib_name])
    # add a dummy to keep the same sequence
    lib_name = np.vstack([np.zeros((1,16855)), lib_name])
    return embed, lib_name

def collate_fn_padd(batch):
    ## padd
    lengths = torch.IntTensor([ embed.shape[0] for embed, _ in batch ]).to(device)
    lengths, perm_index = lengths.sort(0, descending=True)
    embed = torch.nn.utils.rnn.pad_sequence([ torch.Tensor(embed).to(device) for embed, _ in batch ])
    embed = embed[:, perm_index, :]
    lib_name = torch.nn.utils.rnn.pad_sequence([ torch.as_tensor(lib_name, dtype=torch.float, device=device) for _, lib_name in batch ])
    lib_name = lib_name[:, perm_index, :]
    return embed, lib_name, lengths

criterion = nn.BCEWithLogitsLoss(reduction="sum")

def train_clf(embed, lib_name, model, optimizer, lengths):
    model.zero_grad()
    model.train()
    loss = model(embed, lib_name, lengths.cpu(), criterion)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_iters(loader, model, optimizer, step_print=50):
  count = 0
  total = 0
  total_loss = 0
  for embed, lib_name, lengths in loader:
    loss = train_clf(embed, lib_name, model, optimizer, lengths)
    count += 1
    total_loss += loss
    total += 1
    if count % step_print == 0:
      count = 0
      # logger.info("cur loss is {}".format(loss))
  return total_loss / total

def eval(loader, model):
  model.eval()
  total_loss = 0
  total = 0
  for embed, lib_name, lengths in loader:
    with torch.no_grad():
      loss = model(embed, lib_name, lengths.cpu(), criterion)
      total_loss += loss.item()
      total += 1
  return total_loss / total

if __name__ == "__main__":
  cache_data = cacheDataset()
  split_size = int(len(cache_data) * 0.9)
  train_dataset, valid_dataset = random_split(cache_data, [split_size, len(cache_data) - split_size])
  train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)
  save_path = "./clf_saved"


  gen = Generator(768, 768).to(device)
  # gen = torch.load("./gen_consine/best_gen.pt").to(device)
  clf = LibClassifier(gen, 768, 16855).to(device)
  # clf = torch.load("./clf_saved/best_clf.pt").to(device)
  optimizer = torch.optim.Adam(clf.parameters(), lr=2e-5) 

  eval_loss_list = []


  for epoch_no in range(100):
    print("################TRAIN #{} EPOCH################".format(epoch_no))
    train_loss = train_iters(train_loader, clf, optimizer)
    print("train loss is: ", train_loss)
    eval_loss = eval(valid_loader, clf)
    if len(eval_loss_list) == 0 or eval_loss < max(eval_loss_list):
      print("Best eval, saved to disc")
      # torch.save(clf, save_path + "/best_clf.pt")
      torch.save(clf.state_dict(), save_path + "/best_clf_state_dict.pt")
    eval_loss_list.append(eval_loss)
    print("eval loss is: ", eval_loss)
    print("best eval loss is ", max(eval_loss_list))
    # torch.save(clf, save_path + "/last_clf.pt")
    torch.save(clf.state_dict(), save_path + "/last_clf_state_dict.pt")

    


