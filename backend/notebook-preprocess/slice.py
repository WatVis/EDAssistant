  
import subprocess
import os
from os.path import join
import shutil
from tqdm import tqdm, trange
import sys
import multiprocessing.dummy as mp 

notebooks_path = "../notebooks-full"
slice_output_path = "../notebooks-locset"
parse_output_path = "../notebooks-noMD"

devnull = open(os.devnull, 'w')
def sliceNotebooks(in_path, out_path, dir_name):
  '''
  slice notebooks into pieces based on data flow
  '''
  subprocess.run(["node", "./sliceNotebooks.js", in_path, out_path, dir_name], stdout=devnull, stderr=devnull)

def parseNotebooks(in_path, out_path, dir_name):
  '''
  # produce flatten notebook
  '''
  subprocess.run(["node", "./parseNotebooks.js", in_path, out_path, dir_name], stdout=devnull, stderr=devnull)

def mkdirIfNotExists(path):
  if not os.path.exists(path):
    os.mkdir(path)

def parse_dir(idx):
  dir_name = dirnames[idx]
  print("##########################")
  print("parsing competition {}".format(dir_name))
  _, _, filenames = next(os.walk(join(dirpath, dir_name)))
  mkdirIfNotExists(join(slice_output_path, dir_name))
  mkdirIfNotExists(join(parse_output_path, dir_name))
  for idx in tqdm(range(len(filenames)), file=sys.stdout):
    fname = filenames[idx]
    f = fname.split('.')
    if f[1] == 'csv':
      shutil.copyfile(join(dirpath, dir_name, fname), join(slice_output_path, dir_name, fname))
    elif f[1] == 'ipynb':
      sliceNotebooks(join(dirpath, dir_name, fname), join(slice_output_path, dir_name), f[0])
      parseNotebooks(join(dirpath, dir_name, fname), join(parse_output_path, dir_name), f[0])
    else:
      raise RuntimeError("unknown extension {}".format(f[1]))

if __name__=="__main__":
  process_num = 6
  mkdirIfNotExists(slice_output_path)
  mkdirIfNotExists(parse_output_path)
  dirpath, dirnames, _ = next(os.walk(notebooks_path))

  p=mp.Pool(process_num)
  p.map(parse_dir,trange(0,len(dirnames)))
  p.close()
  p.join()