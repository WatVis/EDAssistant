import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Generator
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

  def __len__(self):
    return len(self.embed)

  def __getitem__(self, idx):
    return np.vstack([np.zeros((1,768)), self.embed[idx]])

def collate_fn_padd(batch):
    ## padd
    lengths = torch.IntTensor([ t.shape[0] for t in batch ]).to(device)
    lengths, perm_index = lengths.sort(0, descending=True)
    batch = torch.nn.utils.rnn.pad_sequence([ torch.Tensor(t).to(device) for t in batch ])
    batch = batch[:, perm_index, :]
    return batch, lengths

def criterion_inner(emb1, emb2):
    scores=torch.einsum("ab,cb->ac",emb1,emb2)
    # print(emb1.size(0))
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(scores, torch.arange(emb1.size(0), device=scores.device))
    return loss

loss_func = torch.nn.CosineEmbeddingLoss()

def criterion_cosine(emb1, emb2):
    # scores=torch.einsum("ij,ij->i",emb1,emb2)
    # loss = torch.mean(scores)
    loss = loss_func(emb1, emb2, torch.ones(emb1.size(0)).to(device))
    return loss

def train_gen(data, model, optimizer, lengths):
    model.zero_grad()
    model.train()
    loss = model(data, lengths.cpu(), criterion_cosine)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_iters(loader, model, optimizer, step_print=50):
  count = 0
  total = 0
  total_loss = 0
  for data, lengths in loader:
    loss = train_gen(data, model, optimizer, lengths)
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
  for data, lengths in loader:
    with torch.no_grad():
      loss = model(data, lengths.cpu(), criterion_cosine)
      total_loss += loss.item()
      total += 1
  return total_loss / total

if __name__ == "__main__":
  cache_data = cacheDataset()
  split_size = int(len(cache_data) * 0.9)
  train_dataset, valid_dataset = random_split(cache_data, [split_size, len(cache_data) - split_size])
  train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)
  save_path = "./gen_saved"


  gen = Generator(768, 768).to(device)
  # gen = torch.load(save_path + "/last_gen.pt")
  optimizer_gen = torch.optim.Adam(gen.parameters(), lr=2e-5) 

  eval_loss_list = []


  for epoch_no in range(100):
    print("################TRAIN #{} EPOCH################".format(epoch_no))
    train_loss = train_iters(train_loader, gen, optimizer_gen)
    print("train loss is: ", train_loss)
    eval_loss = eval(valid_loader, gen)
    if len(eval_loss_list) == 0 or eval_loss < min(eval_loss_list):
      print("Best eval, saved to disc")
      # torch.save(gen, save_path + "/best_gen.pt")
      torch.save(gen.state_dict(), save_path + "/best_gen_state_dict.pt")
    eval_loss_list.append(eval_loss)
    print("eval loss is: ", eval_loss)
    print("best eval loss is ", min(eval_loss_list))
    # torch.save(gen, save_path + "/last_gen.pt")
    torch.save(gen.state_dict(), save_path + "/last_gen_state_dict.pt")

    


