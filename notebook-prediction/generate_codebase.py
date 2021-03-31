import numpy as np
from tqdm import trange
embeds = np.load("embed_tensors.npy", allow_pickle=True)
kernel_ids = np.load("kernel_ids.npy", allow_pickle=True)

sort_idx = kernel_ids.argsort()
kernel_ids = kernel_ids[sort_idx]
embeds = embeds[sort_idx]

def parseID(path):
  return path.split('\\')[1].split('_')[0]

notebook_idx = 0
cur_notebook_id = parseID(kernel_ids[0])
embed_list = []
unique_embeds = []
unique_kernelids = []
kid_map = []
for script_idx in trange(len(kernel_ids)):
  kid = parseID(kernel_ids[script_idx])
  embed = embeds[script_idx]
  embed_list.append(embed)
  # dummy bucket
  for i in range(embed.shape[0]):
    kid_map.append('{}##{}'.format(kernel_ids[script_idx], i))
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

np.save('./codebase_embed', np.concatenate(unique_embeds))
np.save('./codebase_id', np.concatenate(unique_kernelids))