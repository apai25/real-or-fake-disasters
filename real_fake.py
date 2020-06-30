import numpy as np 
import pandas as pd

# ====================================TRAINING========================================
# Importing training dataset
train = pd.read_csv('train.csv')

# Creating train_keywords, train_text, and train_targets arrays from training dataset
train_keywords = train.iloc[:, 1].values
train_text = train.iloc[:, 3].values
train_targets = train.iloc[:, -1].values

# Reshaping train_keywords to prepare for imputing
train_keywords = train_keywords.reshape(len(train_keywords), 1)

# Imputing train_keywords to get rid of NaN values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='no_keyword')
train_keywords = imputer.fit_transform(train_keywords)

# One Hot Encoding the train_keywords column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
encoder = OneHotEncoder(dtype=int, drop='first')
ct = ColumnTransformer([('encoding', encoder, [0])], remainder='passthrough')
train_keywords = ct.fit_transform(train_keywords)

# Cleaning the train_text array and creating a corpus
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
train_corpus = []
for i in range(0, len(train_text)):
    tweet = re.sub('[^a-zA-Z]', ' ', train_text[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    all_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if word not in set(all_stopwords)]
    tweet = ' '.join(tweet)
    train_corpus.append(tweet)

# Count Vectorizing train_corpus to create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
train_corpus = cv.fit_transform(train_corpus)

# Transforming train_corpus into an array for concatenation
train_corpus = train_corpus.toarray()

# Reshaping train_keywords into an array for concatenation
train_keywords = train_keywords.toarray()

# Concatenating train_keywords to train_corpus to create x_train array
x_train = np.concatenate((train_keywords, train_corpus), axis=1)

# Creating and training the classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, train_targets)

# ====================================TESTING========================================
# Importing the testing dataset
test = pd.read_csv('test.csv')

# Creating ids, test_keywords, and test_text arrays from testing datsaet
test_keywords = test.iloc[:, 1].values
test_text = test.iloc[:, -1].values
ids = test.iloc[:, 0].values

# Reshaping test_keywords to prepare for imputing
test_keywords = test_keywords.reshape(len(test_keywords), 1)

# Imputing train_keywords to get rid of NaN values
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='no_keyword')
test_keywords = imputer.fit_transform(test_keywords)

# One Hot Encoding test_keywords
test_keywords = ct.transform(test_keywords)

# Cleaning the test_text array and creating a corpus
test_corpus = []
for i in range(0, len(test_text)):
    tweet = re.sub('[^a-zA-Z]', ' ', test_text[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    all_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if word not in set(all_stopwords)]
    tweet = ' '.join(tweet)
    test_corpus.append(tweet)

# Count Vectorizing test_corpus to create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
test_corpus = cv.transform(test_corpus)

# Transforming test_corpus into an array for concatenation
test_corpus = test_corpus.toarray()

# Transforming test_keywords into an array for concatenation
test_keywords = test_keywords.toarray()

# Concatenating test_corpus to test_corpus to create x_test array
x_test = np.concatenate((test_keywords, test_corpus), axis=1)

# Predicting with the classifier
predictions = classifier.predict(x_test)

# Reshaping ids for concatenation
ids = ids.reshape(len(ids), 1)

# Reshaping predictions for concatenation
predictions = predictions.reshape(len(predictions), 1)

# Concatenating ids and predictions
submission = np.concatenate((ids, predictions), axis=1)

# Saving submission to a .csv file
np.savetxt('submission.csv', submission, delimiter=',', fmt='%d')