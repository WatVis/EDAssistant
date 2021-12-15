import pandas as pd
#--------------------CELL--------------------
DATA_PATH = '../input'
#--------------------CELL--------------------
df_train = pd.read_csv(DATA_PATH + '/train.csv', encoding='cp1252')
#--------------------CELL--------------------
df_train['ciphertext_len'] = df_train['ciphertext'].apply(lambda x: len([y.encode() for y in x]))
#--------------------CELL--------------------
df_train.head()
