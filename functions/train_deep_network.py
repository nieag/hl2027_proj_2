import keras
import label_data
import numpy as np
from random import shuffle
from scipy.ndimage import zoom
from keras.models import Sequential, model_from_json
from keras.callbacks import History, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
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
    valid = data[:splitPoint, :, :]
    valid_labels = labels[:splitPoint]

    return train, train_labels, valid, valid_labels


def normalizeData(data, denom=None):
    if not denom:
        denom = np.max(data)
        return data / denom, denom
    else:
        print("denom used")
        return data / denom, denom


def trainModel(x_train, x_valid, y_train, y_valid):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    """
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    """
    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(2))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    hist = History()
    hist = model.fit(x_train, y_train, batch_size=32, nb_epoch=2, validation_data=(x_valid, y_valid), callbacks=[EarlyStopping(patience=2)])

    return model, hist


def saveModel(model, name="simpleCNN"):
    # serialize model to JSON
    model_json = model.to_json()
    with open("functions/dnns/{}/model.json".format(name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("functions/dnns/{}/model.h5".format(name))
    print("Saved model to disk")


def loadModel(name="simpleCNN"):
    json_file = open("functions/dnns/{}/model.json".format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("functions/dnns/{}/model.h5".format(name))
    print("Loaded model from disk")

    return loaded_model


def evaluateModel(model, test_data, test_labels):
    model.evaluate(test_data)


if __name__ == '__main__':
    """
    read data and generate labels from files
    """
    data, labels, test, test_labels = label_data.readLabelsAndFiles()
    """
    Resize the data to save time
    """
    data = resizeData(data)
    test = resizeData(test)
    print("Resized: ", data.shape)
    """
    Shuffle the data to be able to divide in training and valid data without using a single patient for either
    """
    data, labels = shuffleData(data, labels)
    print("Shuffled: ", data.shape)
    """
    normalize data to pixel values 0-1
    """
    data, denom = normalizeData(data)
    test, denom = normalizeData(test, denom)

    print("Normalized: ", data.shape)
    """
    Divide the data into valid/train portions
    """
    x_train, y_train, x_valid, y_valid = splitData(data, labels, p=0.2)
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
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
    test = test.reshape(test.shape[0], test.shape[1], test.shape[2], 1)

    """
    Train a network
    """
    model, history = trainModel(x_train, x_valid, y_train, y_valid)

    evaluateModel(model, test, test_labels)
    """
    Save the model of the network:
    """
    saveModel(model)
    """
    Or load a model:
    """
    # model = loadModel()
