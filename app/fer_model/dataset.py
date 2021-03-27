import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


class Dataset(object):
    def __init__(self, data, labels, data_names, classes):
        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._data_names = data_names
        self._classes = classes
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def data_names(self):
        return self._data_names

    @property
    def classes(self):
        return self._classes

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return (
            self._data[start:end],
            self._labels[start:end],
            self._data_names[start:end],
            self._classes[start:end],
        )


def load_train_images(image_size, fields):
    images = []
    labels = []
    image_names = []
    classes = []

    for field in fields:
        index = fields.index(field)
        files_png = glob.glob("/fer_model/images/training_images/" + field + "/*.png")
        files_tiff = glob.glob("/fer_model/images/training_images/" + field + "/*.tiff")

        files = files_png + files_tiff

        for file in files:
            image = cv2.imread(file)
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(fields))
            label[index] = 1.0
            labels.append(label)
            image_name = os.path.basename(file)
            image_names.append(image_name)
            classes.append(field)

    images = np.array(images)
    labels = np.array(labels)
    image_names = np.array(image_names)
    classes = np.array(classes)
    return images, labels, image_names, classes


def read_train_sets(data_size, classes, data_type):
    class Datasets(object):
        pass

    datasets = Datasets()
    if data_type == "images":
        data, labels, data_names, cls = load_train_images(data_size, classes)
    else:
        data, labels, data_names, cls = load_train_music(classes)
    data, labels, data_names, cls = shuffle(data, labels, data_names, cls)

    validation_size = int(0.2 * data.shape[0])
    validation_data = data[:validation_size]
    validation_labels = labels[:validation_size]
    validation_data_names = data_names[:validation_size]
    validation_classes = cls[:validation_size]

    training_data = data[validation_size:]
    training_labels = labels[validation_size:]
    training_data_names = data_names[validation_size:]
    training_classes = cls[validation_size:]

    datasets.train = Dataset(
        training_data, training_labels, training_data_names, training_classes
    )
    datasets.valid = Dataset(
        validation_data, validation_labels, validation_data_names, validation_classes
    )
    return datasets
