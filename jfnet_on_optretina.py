from __future__ import print_function

import os

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks

import plotting

plotting.allow_plot()

nb_classes = 2
batch_size = 32
nb_epochs = 500
patience = 20
class_weight = {0:1, 1:4.852051234453}

path = '/home/cl/Downloads/OR'
feat_file = 'feat_act_JFnet_21_.npy'
labels_file = 'OR_diseased_labels.csv'

X = np.load(os.path.join(path, feat_file))
labels = pd.read_csv(os.path.join(path, labels_file))
y = labels.diseased.values
nb_classes = len(labels.diseased.unique())
nb_samples, nb_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    stratify=y)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(nb_classes,
                input_dim=nb_features,
                init='glorot_uniform',
                activation='softmax'))

sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


callbacks = []
history = keras.callbacks.History()
callbacks.append(history)

class TrainingMonitor(keras.callbacks.Callback):
    def __init__(self, history):
        super(TrainingMonitor, self).__init__()
        self.loss_plot = plotting.LossPlot(1)
        self.acc_plot = plotting.AccuracyPlot(2)
        self.history = history

    def on_epoch_end(self, epoch, logs={}):
        train_loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]

        train_acc = self.history.history['acc'][-1]
        val_acc = self.history.history['val_acc'][-1]

        self.loss_plot.plot(train_loss, val_loss, epoch)
        self.acc_plot.plot(train_acc, val_acc, epoch)

callbacks.append(TrainingMonitor(history))


if patience:
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   verbose=1,
                                                   mode='auto')
    callbacks.append(early_stopping)


model.fit(X_train, Y_train, nb_epoch=nb_epochs, batch_size=batch_size,
          class_weight=class_weight, validation_split=0.1,
          show_accuracy=True, verbose=1, callbacks=callbacks)

# todo: dump model

test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=batch_size,
                                     show_accuracy=True)

from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import roc_curve, auc
posteriors = model.predict_proba(X_test)
HEALTHY, DISEASED = 0, 1
f_diseased_r, t_diseased_r, thresholds = roc_curve(y_test,
                                                   posteriors[:, DISEASED],
                                                   pos_label=DISEASED)
roc_auc = auc(f_diseased_r, t_diseased_r)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(f_diseased_r, t_diseased_r,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Diseased Rate')
plt.ylabel('True Diseased Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()