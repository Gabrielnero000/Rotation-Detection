# Rotation Detection With Deep Learning

### Dependencies:
* Pandas
* Matplotlib
* Keras

### Instructions to run
* Run `python train.py` to train the model and save it as a `model.h5` file.
* Run `python test.py` to predict the test images, correct them and save them on the `test.corrected` directory. Also generate a CSV with the prediction.

### How it works

The first step was create a reasonable model for the data and after some fine tuning, the final model was generated. It can be found on the `model.py` script.

After that, the dataset was processed using the Keras tools for heavy data augmentation, which can be found on the `dataset.py` script.

So the network was trained with the default Adam optimizer and with categorical crossentropy as loss function, achieving an validation error and accuracy of 0.1061 and a 95.9% respectively.

Finally, the test was done by predicting the test images and using the prediction to correct the rotation of the images and generating an CSV with the prediction of every image.
