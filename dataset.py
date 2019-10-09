from keras.preprocessing.image import ImageDataGenerator

import matplotlib.image as image

import pandas as pd
import numpy as np
import os

# Original data
train_path = 'train.rotfaces/train/'
test_path = 'test.rotfaces/test/'
dataframe = pd.read_csv('train.rotfaces/train.truth.csv')

# Images size
imgs_dim = (64, 64)


# Return the dataset for training
def GetDataset(batch_size):

    # Data augmentation setup
    datagen = ImageDataGenerator(rescale=1/255,
                                 rotation_range=15,
                                 width_shift_range=0.15,
                                 height_shift_range=0.15,
                                 brightness_range=(0.7, 1.3),
                                 shear_range=0.25,
                                 zoom_range=0.15,
                                 channel_shift_range=0.15,
                                 validation_split=0.15)

    train = datagen.flow_from_dataframe(dataframe=dataframe,
                                        directory=train_path,
                                        x_col='fn',
                                        y_col='label',
                                        target_size=imgs_dim,
                                        batch_size=batch_size,
                                        subset='training')

    val = datagen.flow_from_dataframe(dataframe=dataframe,
                                      directory=train_path,
                                      x_col='fn',
                                      y_col='label',
                                      target_size=imgs_dim,
                                      batch_size=batch_size,
                                      subset='validation')

    return (train, val)


# Return the test dataset
def GetTestDataset():

    test_names = sorted(os.listdir(test_path))
    test_imgs = np.ndarray((len(test_names), 64, 64, 3))

    for i in range(len(test_names)):
        test_imgs[i] = (image.imread(
            test_path + test_names[i]) / 255.0).astype(np.float)

    return (test_imgs, test_names)
