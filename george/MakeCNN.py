import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.transform
from skimage import color
from skimage.color import rgb2gray
import os

batch_dim = 32
nr_clase = 62
epochs = 100

def incarca_date(directory_data):
    #parcurgerea de subdirectoare
    dirs = [d for d in os.listdir(directory_data)
            if os.path.isdir(os.path.join(directory_data, d))]
    #creearea de doua liste, imagini si etichete
    imagini = []
    etichete = []
    for d in dirs:
        etichete_dir = os.path.join(directory_data, d)
        file_names = [os.path.join(etichete_dir, f)
                      for f in os.listdir(etichete_dir) if f.endswith(".ppm")]
        for f in file_names:
            imagini.append(skimage.data.imread(f))
            etichete.append(int(d))
    return imagini, etichete

def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

DIR_PATH = "C:\\Program Files\\PyCharm\\Projects\\FasterRCNN"
train_data_dir = os.path.join(DIR_PATH, "TrainingSet")
test_data_dir = os.path.join(DIR_PATH, "TestSet")

imagini_train, etichete_train = incarca_date(train_data_dir)
imagini_test, etichete_test = incarca_date(test_data_dir)

display_images_and_labels(imagini_train, etichete_train)

def resize_data(imag_train):
    imag_data_transform = [skimage.transform.resize(image, (32, 32, 1), mode='constant')
                    for image in imag_train]
    return imag_data_transform

x_train = np.array(resize_data(imagini_train[:5]))
x_test = np.array(resize_data(imagini_test[:5]))

etichete_train = keras.utils.to_categorical(etichete_train, nr_clase)
etichete_test = keras.utils.to_categorical(etichete_test, nr_clase)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 1)))
model.add(Conv2D(64, (3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nr_clase, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])
model.fit(x_train, etichete_train,
          batch_size=batch_dim,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, etichete_test))
scor = model.evaluate(x_test, etichete_test, verbose=0)
print('Loss: ', scor[0])
print('Accuaracy: ', scor[1])