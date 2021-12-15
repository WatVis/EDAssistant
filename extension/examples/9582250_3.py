import pandas as pd
import matplotlib.pyplot as plt
#--------------------CELL--------------------
DATA_PATH = '../input'
#--------------------CELL--------------------
df_train = pd.read_csv(DATA_PATH + '/train.csv', encoding='cp1252')
#--------------------CELL--------------------
df_train['ciphertext_len'] = df_train['ciphertext'].apply(lambda x: len([y.encode() for y in x]))
#--------------------CELL--------------------
y_train = df_train['target']
#--------------------CELL--------------------
diffs = list(range(1, 5))
#--------------------CELL--------------------
from sklearn.model_selection import train_test_split
#--------------------CELL--------------------
def split_idx_by_column(df, column, valid_size=None):
    idxs, idxs_valid = {}, {}
    for d in diffs:
        idx = df.index[df[column] == d]
        if valid_size is None:
            idxs[d] = idx
        else:
            idx, idx_valid = train_test_split(idx, random_state=42, 
                                              test_size=valid_size, stratify=df['target'][idx])
            idxs[d] = idx
            idxs_valid[d] = idx_valid
    if valid_size is None:
        return idxs
    else:
        return idxs, idxs_valid
#--------------------CELL--------------------
train_idxs = split_idx_by_column(df_train, 'difficulty')
train_part_idxs, valid_idxs = split_idx_by_column(df_train, 'difficulty', valid_size=0.1)
#--------------------CELL--------------------
print('train part sizes:', [z.shape[0] for z in train_part_idxs.values()])
#--------------------CELL--------------------
for d in diffs:
    idx = train_part_idxs[d].values
    plt.hist(y_train[idx], bins=20, normed=False, alpha=0.5)
