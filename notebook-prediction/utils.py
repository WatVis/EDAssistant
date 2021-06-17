from parser import DFG_python
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from transformers import RobertaTokenizer
from config import *
import os
import pandas as pd
import numpy as np
import torch
import nbformat
import pickle

# only contain code
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             code_tokens,
             code_ids,
             position_idx,
             dfg_to_code,
             dfg_to_dfg,
             libs_names

    ):
        #The first code function
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg
        self.libs_names=libs_names

class CellFeatures(object):
  def __init__(self, input_features, kernel_id):
    self.input_features = input_features
    self.kernel_id = kernel_id

  def size(self):
    return len(self.input_features)

tokenizer_name='microsoft/graphcodebert-base'
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

LANGUAGE = Language('parser/my-languages.so', 'python')
parser = Parser()
parser.set_language(LANGUAGE) 
parser = [parser,DFG_python]    

# remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
        code_tokens = []
    return code_tokens,dfg   

def convert_examples_to_features(code, libs):
    #extract data flow
    code_tokens,dfg=extract_dataflow(code,parser,"python")
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    code_tokens=code_tokens[:code_length+data_flow_length-2-min(len(dfg),data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:code_length+data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=code_length+data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length    
    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]      
  
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg, libs)

def getMetaData(filename):
  df = pd.read_csv("{}.csv".format(filename))
  meta_list = []
  for index, row in df.iterrows():
    usages = row['USAGE'].split(', ')
    usage_list = []
    for idx in range(0, len(usages), 2):
      usage_list.append('{}.{}'.format(usages[idx], usages[idx+1]))
    meta_list.append((row['CELL'], usage_list))
  return meta_list

def parseNotebook(item):
  filename, kernel_id = item
  cell_list = []
  source = ""
  count = 0
  meta = getMetaData(filename)
  with open('{}.py'.format(filename), encoding='utf-8') as f:
    next, lib_name = meta.pop(0)
    for line in f.readlines():
      # print(count, [line])
      if count == next:
        cell_list.append((source, lib_name))
        if len(meta) != 0:
          next, lib_name = meta.pop(0)
        else:
          # TODO: findout why it happens
          source = ""
          break
        source = ""
      count += 1
      source += line
    if source != "":
      cell_list.append((source, lib_name))
  return (cell_list, kernel_id)

def get_notebook_list(notebooks_path):
  dirpath, dirnames, _ = next(os.walk(notebooks_path))
  file_list = []
  for dir_name in dirnames:
    _, _, filenames = next(os.walk(os.path.join(dirpath, dir_name)))
    for fname in filenames:
      f = fname.split('.')
      if f[1] == 'py':
        file_list.append((os.path.join(dirpath, dir_name, f[0]), dir_name+'\\'+f[0]))
      elif f[1] == 'csv':
        # skip for now
        pass
      else:
        raise RuntimeError("unknown extension {}".format(f[1]))
  return file_list

def get_embedding(code, device, model, libs = None):
  item = convert_examples_to_features(code, libs)
  #calculate graph-guided masked function
  attn_mask=np.zeros((code_length+data_flow_length,
                      code_length+data_flow_length),dtype=np.bool)
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
              
  code_inputs, attn_mask, position_idx = (torch.tensor(item.code_ids).view(1, -1).to(device),
        torch.tensor(attn_mask).view(1, 320, 320).to(device),
        torch.tensor(item.position_idx).view(1, -1).to(device))  
  
  return model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)

def readNotebookAsRaw(competition, kernel_id):
    source_path = '../../kaggle-dataset/notebooks-full/'
    file_path = "{}/{}/{}.ipynb".format(source_path, competition, kernel_id.split('_')[0])
    nb = nbformat.read(file_path, nbformat.NO_CONVERT)
    markdowns = [] # list of markdown for each code-markdown pair
    source = []

    for cell in nb['cells']:
        if (cell['cell_type'] =='markdown'):
            markdowns.append(cell['source'])
        # else code or raw
        elif (cell['cell_type'] =='code'):
            source.extend(filter(lambda x: x != '', cell['source'].split('\n')))
    return source

def readNotebookWithNoMD(competition, kernel_id):
    source_path = '../../kaggle-dataset/notebooks-noMD'
    file_path = "{}/{}/{}.py".format(source_path, competition, kernel_id.split('_')[0])
    with open(file_path) as f:
      source = f.readlines()
    return source

def extractLoc(raw_source, loc_string):
  loc_set = [loc.split(',') for loc in loc_string.split(";")]
  source_list = []
  for loc in loc_set:
    start = int(loc[0])-1
    end = int(loc[2])-1
    if int(loc[3]) != 0:
      end += 1
    source_list.extend(raw_source[start:end])
  return source_list

def combine_features(item):
  index, row = item
  competition = row["competition"]
  kernel_id = row["kernel_id"]
  # source = '\n'.join(extractLoc(readNotebookAsRaw(competition, kernel_id), row['loc']))
  source = ''.join(extractLoc(readNotebookWithNoMD(competition, kernel_id), row['loc']))
  feature = convert_examples_to_features(source, None)
  return feature

def to_embedding(data, model, device):
  code_inputs = data[0].to(device)
  attn_mask = data[1].to(device)
  position_idx = data[2].to(device)  
  with torch.no_grad():
    code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
  return code_vec

