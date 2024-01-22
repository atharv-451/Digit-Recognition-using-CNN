import tensorflow as tf
# # Loading The Datasets
mnist = tf.keras.datasets.mnist

# # Dividing the dataset in to training and test samples
(x_train,y_train),(x_test,y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0], cmap = plt.cm.binary)

# # Checking the values of each pixel
# ## Before Normalization
print(x_train[0])

# # Normalizing the data | Pre-Processing Step
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
plt.imshow(x_train[0], cmap = plt.cm.binary)

# # After Normalization
print(x_train[0])
print(y_train[0])

# # Resizing image to make it suitable for applying Convolution operation
import numpy as np
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE,1)
print("Training Samples dimension",x_trainr.shape)
print("Testing Samples dimension",x_testr.shape)

# # Creating a Deep Neural Network
# ## Training on 60,000 samples of MNIST handwritten dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# # Creating a neural network now
model = Sequential()

## 1st Convolutional Layer 0 1 2 4   (60000,28,28,1)
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:])) ### only for first convolution layer to mention input layer size
model.add(Activation("relu")) ## Activation Function to make it non-linear, <0, remove, >0
model.add(MaxPooling2D(pool_size=(2,2))) ## Maxpooling single maximum value of 2x2,

## 2nd Convolution Layer 
model.add(Conv2D(64, (3,3))) ## 2nd Convolution Layer
model.add(Activation("relu")) ## Activation Function 
model.add(MaxPooling2D(pool_size=(2,2))) ## Maxpooling 

## 3rd Convolution Layer
model.add(Conv2D(64, (3,3))) ## 3rd Convolution Layer
model.add(Activation("relu")) ## Activation Function 
model.add(MaxPooling2D(pool_size=(2,2))) ## Maxpooling 

## Fully Connected Layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

## Fully Connected Layer 2
model.add(Dense(32))
model.add(Activation("relu"))

## Last Fully Connected Layer
model.add(Dense(10)) ##This last layer must be equal to 10
model.add(Activation("softmax"))

model.summary()

print("Total Training Samples = ",len(x_trainr))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics=['accuracy'])

# # Training Model
model.fit(x_trainr,y_train,epochs=5, validation_split = 0.3) ## Training my model

# Evaluating on testing dataset
test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test Loss on 10,000 test samples ",test_loss)
print("Validation Accuracy on 10,000 test samples ",test_acc)

predictions = model.predict([x_testr])

print(predictions)

print(np.argmax(predictions[0]))