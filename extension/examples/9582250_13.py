import pandas as pd
import matplotlib.pyplot as plt
#--------------------CELL--------------------
DATA_PATH = '../input'
#--------------------CELL--------------------
df_train = pd.read_csv(DATA_PATH + '/train.csv', encoding='cp1252')
#--------------------CELL--------------------
df_train['ciphertext_len'] = df_train['ciphertext'].apply(lambda x: len([y.encode() for y in x]))
#--------------------CELL--------------------
X_train_features_sparse = vect.fit_transform(df_train['ciphertext'])
#--------------------CELL--------------------
X_train = X_train_features_sparse.tocsr()
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
print('valid sizes:', [z.shape[0] for z in valid_idxs.values()])
#--------------------CELL--------------------
y_valid_to_concat = []
for d in diffs:
    y_valid_to_concat.append(y_train.loc[valid_idxs[d]])
y_valid = pd.concat(y_valid_to_concat)
y_valid.sort_index(inplace=True)
#--------------------CELL--------------------
for d in diffs:
    idx = valid_idxs[d].values
#--------------------CELL--------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
#--------------------CELL--------------------
from sklearn.linear_model import LogisticRegression
#--------------------CELL--------------------
pipes = {}
for d in diffs:
    pipe = Pipeline(memory=None, steps=[
        ('scaler', MaxAbsScaler(copy=False)),
        ('clf', LogisticRegression(solver='lbfgs', multi_class='multinomial', verbose=2, n_jobs=-1))
    ])
    pipes[d] = pipe
#--------------------CELL--------------------
def train(models, X, y, diff_idxs):
    for d in diffs:
        idx = diff_idxs[d].values
        print(f'difficulty = {d}, samples = {idx.shape[0]}')
        model = models[d]
        model.fit(X[idx], y.loc[idx])
    return models
##%%time
##train(pipes, X_train, y_train, train_part_idxs)
#--------------------CELL--------------------
from sklearn.metrics import confusion_matrix
#--------------------CELL--------------------
def predict(models, X, diff_idxs, show_graph=True, y_truth=None):
    y_preds = {}
    for d in diffs:
        idx = diff_idxs[d].values
        model = models[d]
        y_pred = model.predict(X[idx])
        y_preds[d] = pd.Series(data=y_pred, index=idx)
        print(f'difficulty = {d}, valid_preds = {y_preds[d].shape}')
        if show_graph:
            plt.figure(figsize=(12,4))
            plt.subplot(121)
            plt.title(f'Difficulty {d}')
            plt.hist(y_pred, bins=20, normed=False, label='pred', alpha=0.5)
            if y_truth is not None:
                plt.hist(y_truth[idx], bins=20, label='valid', alpha=0.5)
            plt.gca().set_xticks(range(20))
            plt.grid()
            plt.legend()
            if y_truth is not None:
                cm = confusion_matrix(y_truth[idx], y_pred)
                plt.subplot(122)
                plt.imshow(cm)
                plt.colorbar()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
    y_pred_to_concat = []
    for d in diffs:
        y_pred_to_concat.append(y_preds[d])
    y_pred = pd.concat(y_pred_to_concat)
    y_pred.sort_index(inplace=True)
    return y_pred
#--------------------CELL--------------------
y_valid_pred = predict(pipes, X_train, valid_idxs, y_truth=y_valid)
#--------------------CELL--------------------
from sklearn.metrics import f1_score, precision_recall_fscore_support
#--------------------CELL--------------------
f1_score(y_valid, y_valid_pred, average='macro')
#--------------------CELL--------------------
precision_recall_fscore_support(y_valid, y_valid_pred, average='macro')
#--------------------CELL--------------------
cm = confusion_matrix(y_valid, y_valid_pred)
#--------------------CELL--------------------
df_subm = pd.read_csv(DATA_PATH +'/sample_submission.csv')
df_subm['Predicted'] = y_test_pred
df_subm.head()
