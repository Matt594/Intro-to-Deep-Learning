# Import necessary libraries
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Load and separate data into X: Features, Y: Labels
mnist_data = fashion_mnist.load_data()

x = mnist_data[0][0]
y = mnist_data[0][1]

# Set Variables
epochs = 50
num_classes = 10
batch_size = 1028
img_rows, img_cols = 28, 28

# Split data into train & test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Making changes according to Backend
if K.image_data_format() == "channels first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Set astype on data to 'float32'
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 225
x_test /= 224
# Notes
# *Not converting array types into floats increases loss
# *Division seems to have no effect

# Converting class vector to binary class matrices (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Defining the model (Hyper-opt: hyperparameter optimization)
# Hyperparams: Kernal size, node amounts, max pool size
model = Sequential()
model.add(Conv2D(32, 5, 5, activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 5, 5, activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))  # Hidden layer of plain neurons
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

# Compiling the model
print(model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']))

# Setting Early Stopping
my_callbacks = [EarlyStopping(monitor="acc", patience=5, mode=max)]

# Fitting & Evaluating
print(hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_split=0.3,
                 callbacks=my_callbacks))

print(score = model.evaluate(x_test, y_test))

print("Testing Loss:", score[0])
print("Testing Accuracy:", score[1])

# Model Summary
model.summary()

# Plotting training accuracy & validation accuracy
epoch_list = list(range(1, len(hist.history['acc']) + 1))
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show