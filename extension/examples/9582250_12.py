import pandas as pd
#--------------------CELL--------------------
DATA_PATH = '../input'
#--------------------CELL--------------------
df_test = pd.read_csv(DATA_PATH + '/test.csv', encoding='cp1252')
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
test_idxs = split_idx_by_column(df_test, 'difficulty')
#--------------------CELL--------------------
print('test sizes:', [z.shape[0] for z in test_idxs.values()])
