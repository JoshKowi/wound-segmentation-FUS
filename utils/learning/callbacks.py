# ------------------------------------------------------------ #
#
# file : utils/learning/callbacks.py
# author : CM
# Custom callbacks
#
# ------------------------------------------------------------ #
import keras.src.callbacks
import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from utils.io.data import save_results


# reduce learning rate on each epoch
def learningRateSchedule(initialLr=1e-4, decayFactor=0.99, stepSize=1):
    def schedule(epoch):
        lr = initialLr * (decayFactor ** np.floor(epoch / stepSize))
        print("Learning rate : ", lr)
        return lr
    return LearningRateScheduler(schedule)

class PrintValidations(keras.callbacks.Callback):
    '''Callback to print the models predictions on the same 3 validation-images
    for each epoch.'''

    def __init__(self, dataGenerator):
        super().__init__()
        self.batch_size = 3
        self.val_images , _ = next(dataGenerator.generate_data(batch_size=self.batch_size, val=True))
        self.epoch = 0
        self.preds_directory = f"training_history/val_predictions-{datetime.datetime.now().strftime('%Y-%m-%d--%H-%M')}/"
        os.makedirs(self.preds_directory, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        prediction_names = [f"ep-{self.epoch}-img-{i}.png" for i in range(self.batch_size)] # create filename list
        self.epoch +=1
        prediction = self.model.predict(self.val_images, verbose=1)
        save_results(prediction, 'rgb', self.preds_directory, prediction_names)
