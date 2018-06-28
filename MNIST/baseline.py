# MNIST handwritten digit recognition problem
# Simple NN with one hidden layer

# For plotting ad hoc MNIST instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

# For NN
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


def show_data(X_train):
    # Call within run() function
    # Plot first 4 images of dataset as gray scale
    plt.subplot(221)
    plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    # Show the plot
    plt.show()


def baseline_model(num_pixels, num_classes):
    # Create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run():
    # Set seed for reproducibility
    seed = 7

    numpy.random.seed(seed)
    # Load the MNIST dataset

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Shape and flatten dataset array
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')

    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    # Normalise inputs from 0-255 to 0-1
    X_train = X_train / 255

    X_test = X_test / 255
    # One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    num_classes = y_test.shape[1]

    # Build the model
    model = baseline_model(num_pixels, num_classes)
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline error: %.2f%%" % (100-scores[1]*100))


# Run the script
run()

