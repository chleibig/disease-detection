"""Perform forward pass through network and save extracted features

code is partially inspired by o_O team, see their github repo at
https://github.com/sveitser/kaggle_diabetic

Christian Leibig, 2015

"""
from __future__ import division, print_function

import click
import numpy as np


@click.command()
@click.argument('path_to_images')
@click.argument('csv_file_with_filenames')
@click.option('--batch_size', default=2, show_default=True,
              help="Number of samples to be passed through the network at "
                   "once.")
@click.option('--outfile', default='feat_act_JFnet_LAST_LAYER_.npy',
              show_default=True,
              help="Filename for saving the extracted features.")
def main(path_to_images, csv_file_with_filenames, batch_size, outfile):
    """Perform forward pass through network and save extracted features"""
    import theano
    import theano.tensor as T
    import lasagne

    import models
    from datasets import OptRetina

    input_var = T.tensor4('inputs')
    weights = '/home/cl/Downloads/kdr_solutions/JeffreyDF/' \
              'kaggle_diabetic_retinopathy/dumps/' \
              '2015_07_17_123003_PARAMSDUMP.pkl'
    network = models.jeffrey_df(input_var=input_var, width=512, height=512,
                                filename=weights)
    last_layer = '21'
    output_layer = network[last_layer]

    feature_activations = lasagne.layers.get_output(output_layer)
    forward_pass = theano.function([input_var], feature_activations)

    dataset = OptRetina(path_data=path_to_images,
                        filename_targets=csv_file_with_filenames)
    idx = np.arange(dataset.n_samples)

    n_features = output_layer.output_shape[1]
    assert n_features == 512
    outputs = np.empty((dataset.n_samples, n_features))
    n_batches = np.ceil(dataset.n_samples/batch_size)

    with click.progressbar(dataset.iterate_minibatches(idx, batch_size),
                           label='Forward pass of '+ str(dataset.n_samples)
                                   +' images through network',
                           length=n_batches) as batch_iterator:
        for i, batch in enumerate(batch_iterator):
                inputs, _ = batch
                outputs[i*batch_size:min((i+1)*batch_size,
                        dataset.n_samples)] = forward_pass(inputs)

    print("Writing extracted features to disk...")
    if outfile == 'feat_act_JFnet_LAST_LAYER_.npy':
        outfile = ''.join(['feat_act_JFnet_', last_layer, '_.npy'])
    np.save(outfile, outputs)
    print("Done.")

if __name__ == '__main__':
    main()
