import pandas as pd
import numpy as np
import keras.models
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import itertools


all_data = pd.read_csv('ramhacks data - Sheet1.csv')
#
housing_index_feature_input = all_data['SLOPE']
sentiment_feature_input = all_data['SENTIMENT SCORE']


housing_index_feature = []
sentiment_feature = []

for x in range(245):
    housing_index_feature.append(housing_index_feature_input[x])
for y in range(245):
    sentiment_feature.append(sentiment_feature_input[y])


print(housing_index_feature)
print(sentiment_feature)



training_sentiment_feature = all_data['TESTING SENTIMENT']
training_index_feature = all_data['TESTING SLOPE']



model = Sequential()
model.add(Dense(245, input_dim=1, kernel_initializer='normal', activation='relu'))
# model.add(Dense(80, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=keras.optimizers.sgd(lr=.001, momentum=0.0, decay=0.0, nesterov=False))
#


labels = housing_index_feature
feature_cols = sentiment_feature

model.fit(np.array(feature_cols), np.array(labels), epochs=100, batch_size=5)
#
feature_cols_test = training_sentiment_feature
labels_test = training_index_feature

y = model.predict(np.array(feature_cols_test))
print(y)

