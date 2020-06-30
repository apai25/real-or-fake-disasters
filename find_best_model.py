import numpy as np 
import pandas as pd 

# Importing main dataset
dataset = pd.read_csv('train.csv')

# Creating keywords, text, and target arrays
keywords = dataset.iloc[:, 1].values
text = dataset.iloc[:, 3].values
target = dataset.iloc[:, -1].values

# Reshaping keywords to prepare for imputing
keywords = keywords.reshape(len(keywords), 1)

# Imputing keywords to get rid of NaN values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='no_keyword')
keywords = imputer.fit_transform(keywords)

# One Hot Encoding the keywords array
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
encoder = OneHotEncoder(dtype=int, drop='first')
ct = ColumnTransformer([('encoding', encoder, [0])], remainder='passthrough')
keywords = ct.fit_transform(keywords)

# Cleaning the text array and creating a corpus
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
corpus = []
for i in range(0, len(text)):
    tweet = re.sub('[^a-zA-Z]', ' ', text[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    all_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if word not in set(all_stopwords)]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

# Preparing corpus for concatenation
corpus = np.array(corpus, dtype=str)
corpus = corpus.reshape(len(corpus), 1)

# Preparing keywords for concatenation
keywords = keywords.toarray()

# Creating X array by concatenating keywords and corpus
x = np.concatenate((keywords, corpus), axis=1)

# Reshaping target for splitting
y = target.reshape(len(target), 1)

# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=0)

# Creating two arrays of the text column and count vectorizing them into bag of word models
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
train_text = x_train[:, -1]
train_text = cv.fit_transform(train_text)

test_text = x_test[:, -1]
test_text = cv.transform(test_text)

# Deleting original text columns from x_train, x_test
x_train = np.delete(x_train, -1, 1)
x_test = np.delete(x_test, -1, 1)

# Reshaping train_text, test_text for concatenation
train_text = train_text.toarray()
test_text = test_text.toarray()

# Concatenating count vectorized text column with x_train and x_test
x_train = np.concatenate((x_train, train_text), axis=1)
x_test = np.concatenate((x_test, test_text), axis=1)

# Converting from string into int
x_train = np.array(x_train, dtype=int)
x_test = np.array(x_test, dtype=int)

# Creating, training, testing and printing results of classifier
from test_models import test_classifier
classifier = 'RBF SVM'
test_classifier(classifier, x_train, y_train, x_test, y_test)