"""Perform forward pass through network and save extracted features

code is partially inspired by o_O team, see their github repo at
https://github.com/sveitser/kaggle_diabetic

Christian Leibig, 2015

"""
from __future__ import division
import time
import click
import numpy as np


@click.command()
@click.option('--source_dir', default=None, show_default=True,
              help="Directory with images to be transformed.")
@click.option('--filename_targets', default=None, show_default=True,
              help="Absolute filename of trainLabels.csv")
@click.option('--batch_size', default=256, show_default=True,
              help="Number of samples to be passed through the network at "
                   "once.")
@click.option('--outfile', default='feature_activations.npy',
              show_default=True,
              help="Filename for saving the extracted features.")
@click.option('--last_layer', default='fc7', show_default=True,
              help="Layer up to which features shall be computed.")
def main(source_dir, filename_targets, batch_size, outfile, last_layer):
    import theano
    import theano.tensor as T
    import lasagne

    from modelzoo import vgg19
    from datasets import KaggleDR

    network = vgg19.build_model(load_weights=True)
    output_layer = network[last_layer]
    input_var = T.tensor4('inputs')
    network['input'].input_var = input_var

    feature_activations = lasagne.layers.get_output(output_layer)
    forward_pass = theano.function([input_var], feature_activations)

    kdr = KaggleDR(path_data=source_dir, filename_targets=filename_targets)

    start_time = time.time()
    idx = np.arange(kdr.n_samples)
    print("Loading images from {}...".format(source_dir))
    kdr.load_data(idx)
    print('took {:6.1f} seconds'.format(time.time() - start_time))

    outputs = np.empty((kdr.n_samples, output_layer.num_units))
    i = 0
    n_batches = np.ceil(kdr.n_samples/batch_size)
    print("Computing features of", kdr.n_samples, "sample(s)...")
    start_time = time.time()
    for batch in kdr.iterate_minibatches(idx, batch_size):
            print("Working on batch {}/{}".format(i, n_batches))
            inputs, _ = batch
            outputs[i*batch_size:min((i+1)*batch_size, kdr.n_samples)] = \
                forward_pass(inputs)
            i += 1

    print("Forward pass of", kdr.n_samples, "took",
          np.round((time.time() - start_time), 3), "sec.")
    print("Writing features to disk...")
    np.save(outfile, outputs)
    print("Done.")


if __name__ == '__main__':
    main()
