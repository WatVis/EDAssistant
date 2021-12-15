import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#--------------------CELL--------------------
train_df = pd.read_csv("../input/train.csv")
#--------------------CELL--------------------
train_df = train_df.sample(frac=1)
#--------------------CELL--------------------
train_df['l'] = train_df['ciphertext'].apply(lambda t: len(str(t)))
#--------------------CELL--------------------
train_df.l.describe()
