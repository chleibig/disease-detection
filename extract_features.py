"""Perform forward pass through network and save extracted features

code is partially inspired by o_O team, see their github repo at
https://github.com/sveitser/kaggle_diabetic

Christian Leibig, 2015

"""
from __future__ import division
import time

import click
import numpy as np
import glob


def get_image_files(data_dir, left_only=False):
    fs = glob.glob('{}/*.jpeg'.format(data_dir))
    if left_only:
        fs = [f for f in fs if 'left' in f]
    return np.array(sorted(fs))


@click.command()
@click.option('--source_dir', default=None, show_default=True,
              help="Directory with images to be transformed.")
@click.option('--outfile', default='feature_activations.npy',
              show_default=True,
              help="Filename for saving the extracted features.")
@click.option('--last_layer', default='fc7', show_default=True,
              help="Layer up to which features shall be computed.")
def main(source_dir, outfile, last_layer):
    import theano
    import theano.tensor as T
    import lasagne
    from PIL import Image

    from modelzoo import vgg19
    from datasets import KaggleDR
    ###########################################################################
    # Setup pretrained network: Here VGG19
    ###########################################################################
    network = vgg19.build_model(load_weights=True)
    ###########################################################################
    # Extract features
    output_layer = network[last_layer]

    print("Loading images from {}...".format(source_dir))
    start_time = time.time()
    file_names = get_image_files(source_dir)
    X = np.array([KaggleDR.prepare_image(np.array(Image.open(fn)))
                  for fn in file_names])
    print('took {:6.1f} seconds'.format(time.time() - start_time))

    # Prepare theano variables
    input_var = T.tensor4('inputs')
    # plug symbolic input to network
    network['input'].input_var = input_var

    feature_activations = lasagne.layers.get_output(output_layer)
    forward_pass = theano.function([input_var], feature_activations)

    print("Computing features of", len(X), "sample(s)...")
    start_time = time.time()
    outputs = forward_pass(X)
    print("took", np.round((time.time() - start_time), 3), "sec.")

    np.save(outfile, outputs)


if __name__ == '__main__':
    main()
