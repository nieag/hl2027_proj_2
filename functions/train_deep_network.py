import keras
import label_data
import numpy as np
from random import shuffle
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from keras.models import Sequential, model_from_json
from keras.callbacks import History, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
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


def balanceData(data, labels):
    true_samples = data[labels == 1]
    false_samples = data[labels != 1]

    no_of_true = len(true_samples)
    no_of_false = len(false_samples)

    new_false_sample_indexes = np.random.choice(no_of_false, no_of_true, replace=False)

    new_false_samples = false_samples[new_false_sample_indexes, :, :]

    data_ = np.vstack((true_samples, new_false_samples))

    labels_ = np.hstack(([1] * no_of_true, [0] * no_of_true))

    return np.array(data_), np.array(labels_)


def balanceDataWithDuplicates(data, labels):
    true_samples = data[labels == 1]
    false_samples = data[labels != 1]

    no_of_true = len(true_samples)
    no_of_false = len(false_samples)

    data_ = np.vstack((true_samples, true_samples, false_samples))

    labels_ = np.hstack(([1] * 2 * no_of_true, [0] * no_of_false))

    return np.array(data_), np.array(labels_)


def preprocessData(data, labels):
    num_classes = 2
    """
    Resize the data to save time
    """
    data = resizeData(data)
    """
    normalize data
    """
    data, denom = normalizeData(data)
    """
    Shuffle the data to be able to divide in training and valid data without using a single patient for either
    """
    data, labels = balanceData(data, labels)
    data, labels = shuffleData(data, labels)
    print("Shuffled: ", data.shape)
    """
    The data is heavily one-sided with false labels so we attempt to correct this by randomly removing some false samples
    """

    """
    Divide the data into valid/train portions
    """
    x_train, y_train, x_valid, y_valid = splitData(data, labels, p=0.2)
    print("Split: ", x_train.shape)
    print("partition of labels: ", np.sum(y_valid) / len(y_valid))

    """
    One hot encodig:
    """
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    # test_labels = keras.utils.to_categorical(test_labels, num_classes)
    """
    Keras specific format ?
    """
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
    # test = test.reshape(test.shape[0], test.shape[1], test.shape[2], 1)

    return x_train, x_valid, y_train, y_valid


def loadData():
    my_file = Path("data/data.npy")
    if my_file.exists():
        """
        Load labels and data if they laready exist. This saves some time.
        """
        data = np.load("data/data.npy")
        labels = np.load("data/labels.npy")
        test1 = np.load("data/test1.npy")
        test2 = np.load("data/test2.npy")
        test3 = np.load("data/test3.npy")
        test_labels1 = np.load("data/test_labels1.npy")
        test_labels2 = np.load("data/test_labels2.npy")
        test_labels3 = np.load("data/test_labels3.npy")

    else:
        """
        read data and generate labels from files
        """
        data, labels, test1, test2, test3, test_labels1, test_labels2, test_labels3 = label_data.readLabelsAndFiles()

        np.save("data/data.npy", data)
        np.save("data/labels.npy", labels)
        np.save("data/test1.npy", test1)
        np.save("data/test_labels1.npy", test_labels1)
        np.save("data/test2.npy", test2)
        np.save("data/test_labels2.npy", test_labels2)
        np.save("data/test3.npy", test3)
        np.save("data/test_labels3.npy", test_labels3)
    print("Finished loading data")

    return data, labels, test1, test2, test3, test_labels1, test_labels2, test_labels3


def saveModel(model, history, name="simpleCNN"):
    # serialize model to JSON
    model_json = model.to_json()
    with open("functions/dnns/{}/model.json".format(name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("functions/dnns/{}/model.h5".format(name))
    np.save("functions/dnns/{}/history.npy".format(name), history)

    print("Saved model to disk")


def loadModel(name="simpleCNN"):
    json_file = open("functions/dnns/{}/model.json".format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("functions/dnns/bestWeights/weights.best.hdf")

    loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    history = np.load("functions/dnns/{}/history.npy".format(name))

    print("Loaded model from disk")

    return loaded_model, history


def evaluateModel(model, test_data, test_labels):
    predictions = np.round(model.predict(test_data))

    number_of_samples = len(predictions)

    pred = np.argmax(predictions, 1)
    test_pred = np.argmax(test_labels, 1)

    acc = np.sum(test_pred == pred) / number_of_samples

    print("Accuracy: {}".format(acc))

    return test_pred == pred


def makePrettyPlots(history):
    """
    Plot losses and accuracies of network.
    """
    fig, ax = plt.subplots(2, 1)
    plt.suptitle("Plots of losses and accuracies of network.")
    plt.title("Loss")
    plt.subplot(2, 1, 1)
    plt.plot(history.get('loss'), label="loss")  # training loss
    plt.subplot(2, 1, 1)
    plt.plot(history.get('val_loss'), label="val_loss")  # validation loss
    plt.ylabel("Loss")
    plt.xlabel("epochs")

    plt.legend()

    plt.title("Accuracy")
    plt.subplot(2, 1, 2)
    plt.plot(history.get('acc'), label="acc")  # training acc
    plt.subplot(2, 1, 2)
    plt.plot(history.get('val_acc'), label="val_acc")  # validation acc
    plt.ylabel("Accuracy")
    plt.xlabel("epochs")

    plt.legend()

    fig.tight_layout()

    # save png
    fig.savefig("report/plots/accuracy_shallow.png")

    # save csv
    hist = np.vstack((history.get('loss'), history.get('acc'), history.get('val_loss'), history.get('val_acc')))

    np.savetxt('report/shallow_network_loss_acc.csv', hist.T, delimiter=',')

    #plt.show()


def train_classifier(im_list, labels_list):
    """
    Receive a list of images `im_list` and a list of vectors (one per image) with the labels 0 or 1 depending on the axial 2D slice contains or not the femur head. Returns the trained classifier.
    """
    # Preprocess the data to fit keras models
    x_train, x_valid, y_train, y_valid = preprocessData(np.array(im_list), np.array(labels_list))

    batch_size = 32
    epochs = 30
    input_shape = (x_train.shape[1], x_train.shape[2], 1)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
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

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    print(model.summary())

    hist = History()

    # Augment the data to prevent overfitting
    gen = ImageDataGenerator(rotation_range=4, width_shift_range=0.08, shear_range=0.2,
                           height_shift_range=0.08, zoom_range=0.08)

    gen.fit(x_train)

    test_gen = ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
    valid_generator = test_gen.flow(x_valid, y_valid, batch_size=batch_size)

    filepath = "functions/dnns/simpleCNN/weights.best.hdf"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks = [EarlyStopping(patience=10), checkpoint]

    hist = model.fit_generator(train_generator, nb_epoch=epochs, validation_data=valid_generator, callbacks=callbacks, shuffle=True)

    model.load_weights(filepath)

    return model, hist.history


def femur_head_selection(im, classifier):
    """
    Receive a CT image and the trained classifier. Returns the axial slice number with the maximum probability of containing a femur head.
    """
    im_ = np.array(im)
    im_ = resizeData(im_)
    im_, denom = normalizeData(im_)
    im_ = im_.reshape(im_.shape[0], im_.shape[1], im_.shape[2], 1)

    predictions = classifier.predict(im_)
    probs = predictions[:, 1]
    max_prob = np.argmax(probs)

    print(im[max_prob, :, :].shape)

    return im[max_prob, :, :], probs


if __name__ == '__main__':
    """
    load, format and save data, or if already exists, just load already formatted data:
    """
    print("loading data")
    data, labels, test1, test2, test3, test_labels1, test_labels2, test_labels3 = loadData()
    #classifier, hist = train_classifier(data, labels)

    #saveModel(classifier, hist)
    classifier, hist = loadModel()
    """
    test = np.vstack((test1, test2, test3))
    test = resizeData(test)
    test = test.reshape((test.shape[0], test.shape[1], test.shape[2], 1))

    test_labels = np.hstack((test_labels1, test_labels2, test_labels3))
    test_labels = keras.utils.to_categorical(test_labels, 2)


    evaluateModel(classifier, test, test_labels)
    """

    max_prob_slice1, probs1 = femur_head_selection(test1, classifier)
    max_prob_slice2, probs2 = femur_head_selection(test2, classifier)
    max_prob_slice3, probs3 = femur_head_selection(test3, classifier)

    fig, ax = plt.subplots(figsize=(3, 5))

    ax.imshow(np.max(test1, 2).T, cmap="gray", aspect=0.8)
    ax.plot((1 - probs1) * 512, color="chartreuse")
    plt.ylabel("P(z)")
    plt.xlabel("z")
    plt.ylim(ymax=511)
    plt.yticks([0, 512 / 2, 511], (1, 0.5, 0))
    plt.tight_layout()
    fig.savefig("slice-selected-and-pdf-test1.png")

    fig, ax = plt.subplots(figsize=(3, 5))

    ax.imshow(np.max(test2, 2).T, cmap="gray", aspect=0.80)
    ax.plot((1 - probs2) * 512, color="chartreuse")
    plt.ylabel("P(z)")
    plt.xlabel("z")
    plt.ylim(ymax=511)
    plt.yticks([0, 512 / 2, 511], (1, 0.5, 0))
    plt.tight_layout()
    fig.savefig("slice-selected-and-pdf-test2.png")

    fig, ax = plt.subplots(figsize=(3, 5))

    ax.imshow(np.max(test3, 2).T, cmap="gray", aspect=0.8)
    ax.plot((1 - probs3) * 512, color="chartreuse")
    plt.ylabel("P(z)")
    plt.xlabel("z")
    plt.ylim(ymax=511)
    plt.yticks([0, 512 / 2, 511], (1, 0.5, 0))
    plt.tight_layout()
    fig.savefig("slice-selected-and-pdf-test3.png")

    fig, ax = plt.subplots(1, 3, figsize=(5, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(max_prob_slice1, cmap="gray")
    plt.gca().set_xticklabels([''] * 10)

    plt.subplot(1, 3, 2)
    plt.imshow(max_prob_slice2, cmap="gray")
    plt.gca().set_xticklabels([''] * 10)

    plt.subplot(1, 3, 3)
    plt.imshow(max_prob_slice3, cmap="gray")
    plt.gca().set_xticklabels([''] * 10)

    fig.savefig("three-highest-probs.png")



    # plt.show()
