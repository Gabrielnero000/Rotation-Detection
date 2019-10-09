from dataset import GetTestDataset, GetDataset
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import numpy as np

import os

# Corrected images directory
out_path = 'test.corrected/'

# Load the model from file
model = load_model('model.h5')
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])

# Get the test dataset
(imgs, names) = GetTestDataset()

# Predict the test images
pred = model.predict(imgs)
pred = pred.argmax(axis=-1)

labels = []

if not os.path.exists(out_path):
    os.makedirs(out_path)

# Correct the image rotation based on the prediction and save the corrected image to file
for i in range(len(pred)):
    if pred[i] == 0:
        labels.append('rotated_left')
        imgs[i] = np.rot90(imgs[i], k=3)
    elif pred[i] == 1:
        labels.append('rotated_right')
        imgs[i] = np.rot90(imgs[i])
    elif pred[i] == 2:
        labels.append('upright')
    elif pred[i] == 3:
        labels.append('upside_down')
        imgs[i] = np.rot90(imgs[i], k=2)

    image.imsave(out_path + names[i], imgs[i])

# Save the .csv
data = {'fn': names, 'label': labels}
dataframe = pd.DataFrame.from_dict(data)
dataframe.to_csv('test.preds.csv', index=False)
