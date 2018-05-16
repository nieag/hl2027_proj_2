import keras
import label_data
import numpy as np
from random import shuffle
from scipy.ndimage import zoom
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator


def shuffleData(data, labels=0):
    a = len(labels)
    a = np.arange(a)

    shuffle(a)

    data = data[a, :, :]
    labels = labels[a]

    return data, labels


def resizeData(data, extent=(64, 64)):
    return zoom(data, (1, extent[0] / data.shape[1], extent[1] / data.shape[2]))


def splitData(data, labels, p=0.2):
    splitPoint = int(np.floor(data.shape[0] * p))
    train = data[splitPoint:, :, :]
    train_labels = labels[splitPoint:]
    test = data[:splitPoint, :, :]
    test_labels = labels[:splitPoint]

    return train, train_labels, test, test_labels


def normalizeData(data):
    return data / np.max(data)


if __name__ == '__main__':
    """
    read data and generate labels from files
    """
    data, labels = label_data.readLabelsAndFiles()
    """
    Resize the data to save time
    """
    data = resizeData(data)
    print("Resized: ", data.shape)
    """
    Shuffle the data to be able to divide in training and test data without using a single patient for either
    """
    data, labels = shuffleData(data, labels)
    print("Shuffled: ", data.shape)
    """
    normalize data to pixel values 0-1
    """
    data = normalizeData(data)
    print("Normalized: ", data.shape)
    """
    Divide the data into test/train portions
    """
    x_train, y_train, x_test, y_test = splitData(data, labels, p=0.2)
    print("Split: ", x_train.shape)
    """
    Network settings
    """

    batch_size = 32
    num_classes = 2
    epochs = 100
    input_shape = (data.shape[0], data.shape[1])

    """
    One hot encodig:
    """
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print("X_train: ", x_train.shape)
    print("X_test", x_test.shape)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(2))

    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, nb_epoch=10, validation_data=(x_test, y_test))

    """
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    """
