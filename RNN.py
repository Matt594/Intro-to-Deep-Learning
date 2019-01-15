#Import necessary libraries
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import os

#Suppress Warnings (Optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

#Set Variables
epochs = 3
batch_size = 64

#Load and separate data into X: Features, Y: Labels
top_words = 5000 #Loads top 5000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print(X_train)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

#Defining the model (Hyper-opt: hyperparameter optimization)
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Setting Early Stopping
my_callbacks = [EarlyStopping(monitor="acc", patience=5, mode=max)]

#Fitting & Evaluating
hist = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=1, 
                    validation_split=0.3, 
                    callbacks=my_callbacks)

score = model.evaluate(X_test, y_test)

print("Testing Loss:", score[0])
print("Testing Accuracy:", score[1])

#Model Summary
model.summary()

#Plotting training accuracy & validation accuracy
epoch_list = list(range(1, len(hist.history['acc']) + 1))
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show