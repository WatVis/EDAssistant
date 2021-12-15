import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#--------------------CELL--------------------
train_data = pd.read_csv('../input/train.csv', delimiter=',')
#--------------------CELL--------------------
X = train_data[['Id', 'ciphertext']]
y = train_data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#--------------------CELL--------------------
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import nltk
def Tokenizer(str_input):
    str_input = str_input.lower()
    words = word_tokenize(str_input)
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #stem the words
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words
#--------------------CELL--------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, max_df=0.3, min_df=0.001, max_features=100000)),
    #('svd',   TruncatedSVD(algorithm='randomized', n_components=500)),
    ('clf',   XGBClassifier(objective='multi:softmax', n_estimators=500, num_class=20, learning_rate=0.075, colsample_bytree=0.7, subsample=0.8, eval_metric='merror')),
])
#--------------------CELL--------------------
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=Tokenizer, min_df=0.001, max_df=0.3)
X_tfidf = vectorizer.fit_transform(X.ciphertext)
#--------------------CELL--------------------
from sklearn.decomposition import TruncatedSVD
svd_transformer = TruncatedSVD(algorithm='randomized', n_components=300)
X_svd = svd_transformer.fit_transform(X_tfidf)
#--------------------CELL--------------------
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.075, colsample_bytree=0.7, subsample=0.8)
xgb_classifier.fit(X_svd, y)
#--------------------CELL--------------------
X_test = pd.read_csv('../input/classifying-20-newsgroups-test/test.csv', delimiter=',')
#--------------------CELL--------------------
X_test_tfidf = vectorizer.transform(X_test.ciphertext)
X_test_svd = svd_transformer.transform(X_test_tfidf)
xgb_predictions = xgb_classifier.predict(X_test_svd)
predictions = xgb_predictions
#--------------------CELL--------------------
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, average='weighted'))
print(classification_report(y_test, predictions))
