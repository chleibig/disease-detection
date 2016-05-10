import numpy as np
import theano
import theano.tensor as T
import lasagne
from datasets import KaggleDR
import models

from progressbar import ProgressBar

weights = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'
network = models.jeffrey_df(width=2048, height=2048, filename=weights,
                            untie_biases=False)

X = T.tensor4('inputs')
network['0'].input_var = X
img_dim = T.tensor4('img_dim')
network['22'].input_var = img_dim
conv_combined = lasagne.layers.get_output(network['conv_combined'],
                                          deterministic=True)
probs = lasagne.layers.get_output(network['31'], deterministic=True)

forward_pass = theano.function([X, img_dim], [conv_combined, probs])

# Visual inspection of spatial probs distributions for large images seemed
# a bit more informative for the KaggleDR.standard_normalize trafo instead
# than for the KaggleDR.jf_trafo
kdr = KaggleDR(path_data='data/kaggle_dr/train_JF_2048',
               filename_targets='data/kaggle_dr/trainLabels.csv',
               preprocessing=KaggleDR.standard_normalize)


def build_second_input(n_samples):
    img_height = np.full((n_samples, 1, network['21'].output_shape[2],
                          network['21'].output_shape[3]),
                         inputs.shape[2], dtype=np.float32)
    img_width = np.full((n_samples, 1, network['21'].output_shape[2],
                         network['21'].output_shape[3]),
                        inputs.shape[3], dtype=np.float32)
    img_dim = np.concatenate((img_height, img_width), axis=1)

    return img_dim / 700.  # jeffrey does that under load_image_and_process?!


conv_combined = np.zeros((kdr.n_samples,) +
                         network['conv_combined'].output_shape[1:])
probs = np.zeros((kdr.n_samples,) + network['31'].output_shape[1:])
batch_size = 2
progbar = ProgressBar(int(kdr.n_samples / batch_size))

for idx, batch in enumerate(kdr.iterate_minibatches(
                            np.arange(kdr.n_samples), batch_size=batch_size,
                            shuffle=False)):
    inputs, targets = batch
    n_s = len(targets)
    _img_dim = build_second_input(n_s)
    current = slice(idx * batch_size, idx * batch_size + n_s)
    conv_combined[current], probs[current] = forward_pass(inputs, _img_dim)
    progbar.update_iteration(idx)

np.savez_compressed('experiments/train_JF_2048_JFnet_feat.npz',
                    conv_combined=conv_combined,
                    probs=probs)
