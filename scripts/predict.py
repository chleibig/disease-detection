from __future__ import print_function
import click


@click.command()
@click.option('--mc_samples', '-s', default=100, show_default=True,
              help="Number of MC dropout samples, usually called T.")
@click.option('--dataset', '-d', default='KaggleDR_test', show_default=True,
              help="Choose out of: ['KaggleDR_test', 'KaggleDR_train',"
                   "'Messidor', 'Messidor_R0vsR1'] or provide a path to"
                   "images")
@click.option('--preprocessing', '-p', default='JF', show_default=True,
              help="Choose out of: ['JF', 'JF_BG']")
@click.option('--normalization', '-n', default='jf_trafo', show_default=True,
              help="Choose out of: ['jf_trafo', 'standard_normalize']")
@click.option('--model', '-m', default='JFnet', show_default=True,
              help="String 'JFnet' or a pickle file from models.save_model")
@click.option('--batch_size', '-b', default=512, show_default=True)
@click.option('--out_file', '-f',
              default='{mc_samples}_mc_{dataset}_{model}.pkl',
              show_default=True)
def main(mc_samples, dataset, preprocessing, normalization, model,
         batch_size, out_file):
    """Perform and save stochastic forward passes"""

    import cPickle as pickle
    import os

    import numpy as np
    import pandas as pd
    import theano
    from keras.utils.generic_utils import Progbar

    import models
    from datasets import KaggleDR, Messidor, DatasetFromDirectory

    if dataset == 'KaggleDR_test':
        images = 'test_' + preprocessing + '_512'
        labels = 'data/kaggle_dr/retinopathy_solution_wh.csv'
        ds = KaggleDR(path_data=os.path.join('data/kaggle_dr', images),
                      filename_targets=labels,
                      preprocessing=getattr(KaggleDR, normalization))
    elif dataset == 'KaggleDR_train':
        images = 'train_' + preprocessing + '_512'
        labels = 'data/kaggle_dr/trainLabels_wh.csv'
        ds = KaggleDR(path_data=os.path.join('data/kaggle_dr', images),
                      filename_targets=labels,
                      preprocessing=getattr(KaggleDR, normalization))
    elif 'Messidor' in dataset:
        images = 'data/messidor/' + preprocessing + '_512'
        if dataset == 'Messidor':
            labels = 'data/messidor/messidor_wh.csv'
        elif dataset == 'Messidor_R0vsR1':
            labels = 'data/messidor/messidor_R0vsR1.csv'
        else:
            print('Unknown dataset, aborting.')
        ds = Messidor(path_data=images,
                      filename_targets=labels,
                      preprocessing=getattr(KaggleDR, normalization))
    else:
        print('Constructing a dataset from the images in:', dataset)
        ds = DatasetFromDirectory(path_data=dataset,
                                  preprocessing=getattr(KaggleDR,
                                                        normalization))

    if model == 'JFnet':
        model_name = model
        model = models.JFnet(width=512, height=512)
        df = pd.read_csv(labels)
        width = df.width.values.astype(theano.config.floatX)
        height = df.height.values.astype(theano.config.floatX)
    else:
        assert model.endswith('.pkl'), 'model is not a pickle file.'
        model_name = model.split('.pkl')[0]
        model = models.load_model(model)

    n_out = model.net.values()[-1].output_shape[1]

    det_out = np.zeros((ds.n_samples, n_out), dtype=np.float32)
    stoch_out = np.zeros((ds.n_samples, n_out, mc_samples), dtype=np.float32)

    idx = 0
    progbar = Progbar(ds.n_samples)
    for X, _ in ds.iterate_minibatches(np.arange(ds.n_samples),
                                       batch_size=batch_size,
                                       shuffle=False):

        n_s = X.shape[0]
        if isinstance(model, models.JFnet):
            img_dim = models.JFnet.get_img_dim(width[idx:idx + n_s],
                                               height[idx:idx + n_s])
            inputs = [X, img_dim]
        else:
            inputs = [X]

        det_out[idx:idx + n_s] = model.predict(*inputs)
        stoch_out[idx:idx + n_s] = model.mc_samples(*inputs,
                                                    T=mc_samples)
        idx += n_s
        progbar.add(n_s)

    results = {'det_out': det_out,
               'stoch_out': stoch_out}

    if out_file == '{mc_samples}_mc_{dataset}_{model}.pkl':
        out_file = out_file.format(mc_samples=mc_samples,
                                   dataset=dataset,
                                   model=model_name)
    with open(out_file, 'wb') as h:
        pickle.dump(results, h)


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    main()
