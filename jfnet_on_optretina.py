from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import keras.callbacks
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU

import plotting
plotting.allow_plot()
# plotting.close_all()

from util import AdaptiveLearningRateScheduler
from util import TrainingMonitor

import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

nb_classes = 2
batch_size = 256
nb_epochs = 1000
patience = 30
lr_decay = 0.5
lr_patience = 30
l2_lambda = 0.005
undersample = False
# class_weight = {0:1, 1:4.852051234453}
class_weight = {0:1, 1:100}
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

path = '/home/cl/optretina/data'
images = os.path.join(path, 'data_JF_512')
feat_file = 'feat_act_JFnet_21_.npy'
labels_file = 'OR_diseased_labels.csv'

exclude_path = '/home/cl/optretina/data/data_JF_512_exclude'

df = pd.read_csv(os.path.join(path, labels_file))

def include_indices(exclude_path, df):
    exclude_filenames = [fn.split('.')[0] for fn in os.listdir(exclude_path)]

    images = df.filename.apply(lambda fn: fn.split('.')[0]).values
    centre_ids = df.centre_id.values.astype(str)
    filenames = pd.Series(['_'.join(centre_id_and_image) for
                           centre_id_and_image in
                           zip(centre_ids, images)])

    return filenames.index[~filenames.isin(exclude_filenames)]

include_idx = include_indices(exclude_path, df)

X = np.load(os.path.join(path, feat_file))[include_idx]
y = df.diseased.values[include_idx]
n_images_total = len(df)
df = df.iloc[include_idx]
n_images_accepted = len(df)

print('Excluding {} out of {} images.'.format(n_images_total -
                                              n_images_accepted,
                                              n_images_total))
print('Relative class frequencies: ', pd.value_counts(y)/float(len(y)))

nb_classes = len(np.unique(y))
nb_samples, nb_features = X.shape

def undersample_indices(y):
    classes = np.unique(y)
    n_samples_k = pd.value_counts(y).min()
    idx_instances = [np.where(y == k)[0] for k in classes]
    # for k in classes:
    #     np.random.shuffle(idx_instances[k])
    idx_undersample = np.concatenate(tuple([idx_instances[k][:n_samples_k]
                                            for k in classes]), axis=0)
    return idx_undersample

# idx = undersample_indices(y)
# X = X[idx]
# y = y[idx]
# df = df.iloc[idx]

X_train, X_test, y_train, y_test, df_train, df_test  = \
                   train_test_split(X, y, df, test_size=0.1, stratify=y)

if undersample:
    priors_train = {k: np.count_nonzero(y_train == k)/float(len(y_train))
                    for k in np.unique(y_train)}
    idx = undersample_indices(y_train)
    X_train = X_train[idx]
    y_train = y_train[idx]
    df_train = df_train.iloc[idx]
    print('Relative class frequencies of training and validation data due to'
          ' undersampling:', pd.value_counts(y_train)/float(len(y_train)))
else:
    priors_train = {k: 1.0/nb_classes for k in np.unique(y_train)}

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_dim=nb_features, init='glorot_normal',
                activation=LeakyReLU(alpha=0.3),
                W_regularizer=l2(l2_lambda)))
model.add(Dense(512, input_dim=512, init='glorot_normal',
                activation=LeakyReLU(alpha=0.3),
                W_regularizer=l2(l2_lambda)))
model.add(Dense(nb_classes,
                input_dim=512,
                init='glorot_normal',
                activation='softmax',
                W_regularizer=l2(l2_lambda)))

model.compile(loss='binary_crossentropy', optimizer=optimizer)

callbacks = []
history = keras.callbacks.History()
callbacks.append(history)
callbacks.append(TrainingMonitor(history))
if patience:
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   verbose=1,
                                                   mode='auto')
    callbacks.append(early_stopping)

initial_lr = float(optimizer.lr.get_value())
callbacks.append(AdaptiveLearningRateScheduler(initial_lr=initial_lr,
                                               decay=lr_decay,
                                               patience=lr_patience,
                                               verbose=1))

cross_entropy_train  = model.evaluate(X_train, Y_train)
cross_entropy_chance = - np.log(1.0/nb_classes)
print('Cross entropy for chance level:', cross_entropy_chance)
print('Cross entropy before training:', cross_entropy_train)

model.fit(X_train, Y_train, nb_epoch=nb_epochs, batch_size=batch_size,
          class_weight=class_weight, validation_split=0.2,
          show_accuracy=True, verbose=1, callbacks=callbacks)

# todo: dump model

from sklearn.metrics import confusion_matrix
# p_given_x = model.predict_proba(X_test)
# posteriors = np.r_[[p_given_x[:, k] * priors_train[k] for k in priors_train]].T
posteriors = model.predict_proba(X_test)
df_test.loc[:,'pred'] = model.predict_classes(X_test)
cm = confusion_matrix(y_test, df_test.pred)

from sklearn.metrics import roc_curve, auc

HEALTHY, DISEASED = 0, 1
df_test.loc[:, 'prob_healthy'] = posteriors[:, HEALTHY]
df_test.loc[:, 'prob_diseased'] = posteriors[:, DISEASED]
f_diseased_r, t_diseased_r, thresholds = roc_curve(y_test,
                                                   posteriors[:, DISEASED],
                                                   pos_label=DISEASED)

roc_auc = auc(f_diseased_r, t_diseased_r, reorder=True)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(f_diseased_r, t_diseased_r,
         label='ROC curve (auc = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Diseased Rate')
plt.ylabel('True Diseased Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, posteriors[:,
                                                        DISEASED],
                                              pos_label=DISEASED)
pr_auc = auc(recall, precision, reorder=False)
# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision, label='PR curve (auc= %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision recall curve')
plt.legend(loc="lower left")
plt.show()

df_test.to_csv('test_predictions.csv', index=False)