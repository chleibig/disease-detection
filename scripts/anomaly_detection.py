from __future__ import print_function
import cPickle as pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

plt.ion()
sns.set_context('paper', font_scale=1.5)
sns.set_style('whitegrid')

path = 'data/processed/'

configs = {1: {'kaggle_out': '100_mc_KaggleDR_test_BayesJFnet17_392bea6.pkl',
               'imagenet_out': '100_mc_imagenet_val_BayesJFnet17_392bea6.pkl',
               'messidor_out': '100_mc_Messidor_BayesJFnet17_392bea6.pkl',
               'kaggle_feat_train':
                   '0_mc_KaggleDR_train_BayesJFnet17_global_pool_392bea6.pkl',
               'kaggle_feat_test':
                   '0_mc_KaggleDR_test_BayesJFnet17_global_pool_392bea6.pkl',
               'messidor_feat':
                   '0_mc_Messidor_BayesJFnet17_global_pool_392bea6.pkl',
               'imagenet_feat':
                   '0_mc_imagenet_val_BayesJFnet17_global_pool_392bea6.pkl',
               'autoencoder': 'ae_1.h5',
               'labels': 'data/kaggle_dr/trainLabels_bin.csv'},
           2: {'kaggle_out': '100_mc_KaggleDR_test_bcnn2_b69aadd.pkl',
               'imagenet_out':
                   '100_mc_imagenet_val_BayesianJFnet17_onset2_b69aadd.pkl',
               'messidor_out':
                   '100_mc_Messidor_BayesianJFnet17_onset2_b69aadd.pkl',
               'kaggle_feat_train':
                   '0_mc_KaggleDR_train_bcnn2_b69aadd_global_pool.pkl',
               'kaggle_feat_test':
                   '0_mc_KaggleDR_test_bcnn2_b69aadd_global_pool.pkl',
               'messidor_feat':
                   '0_mc_Messidor_bcnn2_b69aadd_global_pool.pkl',
               'imagenet_feat':
                   '0_mc_imagenet_val_bcnn2_b69aadd_global_pool.pkl',
               'autoencoder': 'ae_2.h5',
               'labels': 'data/kaggle_dr/trainLabels_01vs234.csv'}
           }


def load_uncertainties(config):
    with open(os.path.join(path, config['kaggle_out']), 'rb') as h:
        pred_kaggle_out = pickle.load(h)
    with open(os.path.join(path, config['messidor_out']), 'rb') as h:
        pred_messidor_out = pickle.load(h)
    with open(os.path.join(path, config['imagenet_out']), 'rb') as h:
        pred_imagenet_out = pickle.load(h)
    std_kaggle_out = pred_kaggle_out['stoch_out'].std(axis=-1)[:, 1]
    std_messidor_out = pred_messidor_out['stoch_out'].std(axis=-1)[:, 1]
    std_imagenet_out = pred_imagenet_out['stoch_out'].std(axis=-1)[:, 1]
    return std_kaggle_out, std_messidor_out, std_imagenet_out


def load_features(filename):
    with open(os.path.join(path, filename), 'rb') as h:
        feat = pickle.load(h)['det_out']
    return feat


def autoencoder(config):
    filepath = os.path.join(path, config['autoencoder'])
    if os.path.exists(filepath):
        print('Loading model %s' % filepath)
        autoencoder = keras.models.load_model(filepath)
        return autoencoder

    print('Training %s from scratch...' % filepath)
    input_feat = Input(shape=(512,))
    encoded = Dense(128, activation='relu')(input_feat)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(512, activation='linear')(decoded)

    autoencoder = Model(input=input_feat, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')

    X_train = load_features(config['kaggle_feat_train'])
    y_train = pd.read_csv(config['labels']).level.values
    X_train_tmp, X_val_tmp = train_test_split(X_train,
                                              test_size=0.1,
                                              stratify=y_train)
    callbacks = [ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 save_best_only=True),
                 EarlyStopping(patience=5, verbose=1)]

    autoencoder.fit(X_train_tmp, X_train_tmp,
                    nb_epoch=500,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_val_tmp, X_val_tmp),
                    verbose=1,
                    callbacks=callbacks)

    return autoencoder


def squared_reconstruction_error(autoencoder, X):
    return ((autoencoder.predict(X) - X)**2).mean(axis=1)


def uncertainty_plot(config):
    std_kaggle_out, std_messidor_out, std_imagenet_out = \
        load_uncertainties(config)
    sns.kdeplot(std_kaggle_out, shade=True, cut=3, label='Kaggle')
    sns.kdeplot(std_messidor_out, shade=True, cut=3, label='Messidor')
    sns.kdeplot(std_imagenet_out, shade=True, cut=3, label='Imagenet')
    plt.xlabel('uncertainty [$\sigma_{pred}$]')
    plt.ylabel('density [a.u.]')
    plt.xlim(0)
    plt.legend(loc='best')


def anomaly_plot(config):
    ae = autoencoder(config)
    X_kaggle = load_features(config['kaggle_feat_test'])
    X_messidor = load_features(config['messidor_feat'])
    X_imagenet = load_features(config['imagenet_feat'])
    sns.kdeplot(squared_reconstruction_error(ae, X_kaggle),
                shade=True, cut=3, label='Kaggle')
    sns.kdeplot(squared_reconstruction_error(ae, X_messidor),
                shade=True, cut=3, label='Messidor')
    sns.kdeplot(squared_reconstruction_error(ae, X_imagenet),
                shade=True, cut=3, label='Imagenet')
    plt.xlabel('anomaly score [$\overline{(x - x_{reconstructed})^2}$]')
    plt.ylabel('density [a.u.]')
    plt.xlim(0)
    plt.ylim(0, 1500)
    plt.xlim(0, 0.003)


def figure():
    ax221 = plt.subplot(221)
    ax221.set_title('(a) Disease onset: mild DR', loc='left')
    uncertainty_plot(configs[1])

    ax222 = plt.subplot(222)
    ax222.set_title('(b) Disease onset: moderate DR', loc='left')
    uncertainty_plot(configs[2])

    ax223 = plt.subplot(223)
    ax223.set_title('(c) Disease onset: mild DR', loc='left')
    anomaly_plot(configs[1])

    ax224 = plt.subplot(224)
    ax224.set_title('(d) Disease onset: moderate DR', loc='left')
    anomaly_plot(configs[2])


if __name__ == '__main__':
    figure()
