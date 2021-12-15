import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#--------------------CELL--------------------
X_test = pd.read_csv('../input/classifying-20-newsgroups-test/test.csv', delimiter=',')
X_test.head()
