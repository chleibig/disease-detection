from __future__ import print_function
import click


@click.command()
@click.option('--mc_samples', default=100, show_default=True,
              help="Number of MC dropout samples, usually called T.")
@click.option('--dataset', default='KaggleDR_test', show_default=True,
              help="Choose out of: ['KaggleDR_test', 'KaggleDR_train']")
@click.option('--preprocessing', default='JF', show_default=True,
              help="Choose out of: ['JF', 'JF_BG']")
@click.option('--normalization', default='jf_trafo', show_default=True,
              help="Choose out of: ['jf_trafo', 'standard_normalize']")
@click.option('--model', default='JFnet', show_default=True,
              help="String 'JFnet' or a pickle file from models.save_model")
@click.option('--batch_size', default=512, show_default=True)
@click.option('--out_file', default='{mc_samples}_mc_{dataset}_{model}.pkl',
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
    from datasets import KaggleDR
    from util import quadratic_weighted_kappa

    if model == 'JFnet':
        model_name = model
        model = models.JFnet(width=512, height=512)
    else:
        assert model.endswith('.pkl'), 'model is not a pickle file.'
        model_name = model.split('.pkl')[0]
        model = models.load_model(model)

    if dataset == 'KaggleDR_test':
        images = 'test_' + preprocessing + '_512'
        labels = 'data/kaggle_dr/retinopathy_solution_wh.csv'
        ds = KaggleDR(path_data=os.path.join('data/kaggle_dr', images),
                      filename_targets=labels,
                      preprocessing=getattr(KaggleDR, normalization))
        df = pd.read_csv(labels)
        width = df.width.values.astype(theano.config.floatX)
        height = df.height.values.astype(theano.config.floatX)
    elif dataset == 'KaggleDR_train':
        images = 'train_' + preprocessing + '_512'
        labels = 'data/kaggle_dr/trainLabels_wh.csv'
        ds = KaggleDR(path_data=os.path.join('data/kaggle_dr', images),
                      filename_targets=labels,
                      preprocessing=getattr(KaggleDR, normalization))
        df = pd.read_csv(labels)
        width = df.width.values.astype(theano.config.floatX)
        height = df.height.values.astype(theano.config.floatX)
    else:
        print('Unknown dataset, aborting.')
        return

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

    det_y_pred = np.argmax(det_out, axis=1)
    det_acc = np.mean(np.equal(det_y_pred, ds.y))
    det_kappa = quadratic_weighted_kappa(det_y_pred, ds.y, n_out)

    results = {'det_out': det_out,
               'stoch_out': stoch_out,
               'det_kappa': det_kappa,
               'det_acc': det_acc}

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
