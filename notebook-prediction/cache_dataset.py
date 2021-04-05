import argparse
import logging
import pickle
import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import multiprocessing
from tree_sitter import Language, Parser
from utils import CellFeatures, convert_examples_to_features, parseNotebook, get_notebook_list

def combine_features(item):
  cell_list, kernel_id = item
  feautres = []
  for cell, libs in cell_list:
    feautres.append(convert_examples_to_features(cell, libs))
  return CellFeatures(feautres, kernel_id)

if __name__ == "__main__":
  print("start parsing...")
  cpu_cont = 6
  file_path = '../../kaggle-dataset/sliced-notebooks-full-new'
  pool = multiprocessing.Pool(cpu_cont)
  flist = get_notebook_list(file_path)
  data=[]
  data=pool.map(parseNotebook, tqdm(flist, total=len(flist)))
  print("start combining...")
  cache_file='./test.pkl'
  examples = []
  examples=pool.map(combine_features, tqdm(data,total=len(data)))
  pickle.dump(examples,open(cache_file,'wb'))
