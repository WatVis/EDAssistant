  
import subprocess
import os
from os.path import join
import shutil
from tqdm import tqdm 
import sys

# node .\src\index.js 10269993 .\examples\
def parse_file(in_path, out_path, dir_name):
  print("parsing {} to {}/{}.......".format(in_path, out_path, dir_name), flush = True)
  subprocess.run(["node", ".\helloworld.js", in_path, out_path, dir_name])

# os.mkdir("..\kaggle-examples")
# parse_file("10269993", ".\examples", "..\kaggle-examples")

notebooks_path = "../kaggle-dataset/notebooks-full"

output_path = "../kaggle-dataset/sliced-notebooks-full-new"

os.mkdir(output_path)

dirpath, dirnames, _ = next(os.walk(notebooks_path))

for dir_name in dirnames:
  print("##########################")
  print("parsing competition {}".format(dir_name))
  _, _, filenames = next(os.walk(join(dirpath, dir_name)))
  os.mkdir(join(output_path, dir_name))
  for idx in tqdm(range(len(filenames)), file=sys.stdout):
    fname = filenames[idx]
    f = fname.split('.')
    if f[1] == 'csv':
      shutil.copyfile(join(dirpath, dir_name, fname), join(output_path, dir_name, fname))
    elif f[1] == 'ipynb':
      parse_file(join(dirpath, dir_name, fname), join(output_path, dir_name), f[0])
    else:
      raise RuntimeError("unknown extension {}".format(f[1]))
  break