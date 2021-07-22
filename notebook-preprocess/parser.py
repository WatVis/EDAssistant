  
import subprocess
import os
from os.path import join
import shutil
from tqdm import tqdm, trange
import sys
import multiprocessing.dummy as mp 

# node .\src\index.js 10269993 .\examples\
devnull = open(os.devnull, 'w')
def parse_file(in_path, out_path, dir_name):
  #print("parsing {} to {}/{}.......".format(in_path, out_path, dir_name), flush = True)
  subprocess.run(["node", "./helloworld.js", in_path, out_path, dir_name], stdout=devnull, stderr=devnull)

notebooks_path = "../../kaggle-dataset/notebooks-full"

output_path = "../../kaggle-dataset/notebooks-locset-fake"

os.mkdir(output_path)

dirpath, dirnames, _ = next(os.walk(notebooks_path))

def parse_dir(idx):
  dir_name = dirnames[idx]
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

if __name__=="__main__":
  p=mp.Pool(5)
  p.map(parse_dir,trange(0,len(dirnames)))
  p.close()
  p.join()
  # parse_dir(dirnames.index('extra_kaggle'))