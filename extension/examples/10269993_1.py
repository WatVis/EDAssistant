import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#--------------------CELL--------------------
train_data = pd.read_csv('../input/train.csv', delimiter=',')
train_data.head()
