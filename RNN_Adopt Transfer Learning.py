import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train_modified.csv', delimiter = '\t', quoting = 3, header = None)
dataset = dataset.iloc[:,-1].str.split(",", n = 1, expand = True)

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize 
corpus = []
for i in range(0, 10000):
  review = re.sub('[^a-zA-Z]', ' ', dataset[1][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  all_stopwords.remove("don't")
  all_stopwords.remove("didn't")
  all_stopwords.remove('ain')
  all_stopwords.remove("couldn't")
  all_stopwords.remove("haven't")
  all_stopwords.remove("isn't")
  all_stopwords.remove("shouldn't")
  all_stopwords.remove("won't")
  all_stopwords.remove("wouldn't")
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  #review = word_tokenize(review)
  corpus.append(review)
  
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5400)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:10000, 0].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# from keras.preprocessing import sequence

# max_words = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_words)
# X_test = sequence.pad_sequences(X_test, maxlen=max_words)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

regressor.summary()

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 10, batch_size = 50)

# from keras import Sequential
# from keras.layers import Embedding, LSTM, Dense, Dropout
# vocabulary_size = 5000
# embedding_size=32
# model=Sequential()
# model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))

# print(model.summary())


# model.compile(loss='categorical_crossentropy', 
#               optimizer='rmsprop', 
#               metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 10, batch_size = 32)

new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = model.predict(new_X_test)
print(new_y_pred)


new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = model.predict(new_X_test)
print(new_y_pred)