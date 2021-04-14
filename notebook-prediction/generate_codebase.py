import numpy as np
from tqdm import trange
import pickle
import os
from os.path import join

# lib_names = np.load("lib_names.npy", allow_pickle=True)

# lib_list = []
# for lib_docs in lib_names:
#   for lib_cells in lib_docs:
#     lib_list.append(lib_cells)
# lib_unique = np.unique(np.concatenate(lib_list))
# lib_dict = {}
# for idx in trange(lib_unique.shape[0]):
#   lib_dict[lib_unique[idx]] = idx

# pickle.dump(lib_dict, open("lib_dict.pkl",'wb'))

embeds = np.load("embed_tensors.npy", allow_pickle=True)
kernel_ids = np.load("kernel_ids.npy", allow_pickle=True)

sort_idx = kernel_ids.argsort()
kernel_ids = kernel_ids[sort_idx]
embeds = embeds[sort_idx]

def parseID(path):
  # return path.split('\\')[1].split('_')[0]
  return path.split('\\')[0]

notebook_idx = 0
cur_notebook_id = parseID(kernel_ids[0])
embed_list = []
unique_embeds = []
unique_kernelids = []
kid_map = []
for script_idx in trange(len(kernel_ids)):
  kid = parseID(kernel_ids[script_idx])
  
  if kid != cur_notebook_id:
    notebook_idx += 1
    cur_notebook_id = kid
    embed_arr = np.concatenate(embed_list)
    new_embeds, unq_idx = np.unique(embed_arr, axis=0, return_index=True)
    new_kids = np.array(kid_map)
    new_kids = new_kids[unq_idx]
    unique_embeds.append(new_embeds)
    unique_kernelids.append(new_kids)
    embed_list = []
    kid_map = []
  
  embed = embeds[script_idx]
  embed_list.append(embed)
  for i in range(embed.shape[0]):
    kid_map.append('{}##{}'.format(kernel_ids[script_idx], i))

# last competition
notebook_idx += 1
embed_arr = np.concatenate(embed_list)
new_embeds, unq_idx = np.unique(embed_arr, axis=0, return_index=True)
new_kids = np.array(kid_map)
new_kids = new_kids[unq_idx]
unique_embeds.append(new_embeds)
unique_kernelids.append(new_kids)

print("total:", notebook_idx)

np.save('./codebasev2_embed', np.concatenate(unique_embeds))
np.save('./codebasev2_id', np.concatenate(unique_kernelids))