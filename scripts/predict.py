from __future__ import print_function
import click


@click.command()
@click.option('--mc_samples', default=100, show_default=True,
              help="Number of MC dropout samples, usually called T.")
@click.option('--dataset', default='KaggleDR_test', show_default=True,
              help="Choose out of: ['KaggleDR_test', 'KaggleDR_train']")
@click.option('--batch_size', default=512, show_default=True)
@click.option('--out_file', default='{mc_samples}_mc_{dataset}_JFnet.pkl',
              show_default=True)
def main(mc_samples, dataset, batch_size, out_file):
    """Perform and save stochastic forward passes"""

    import cPickle as pickle

    import numpy as np
    import pandas as pd
    import theano
    from keras.utils.generic_utils import Progbar

    from datasets import KaggleDR
    from models import JFnet
    from util import quadratic_weighted_kappa

    jfmodel = JFnet(width=512, height=512)

    if dataset == 'KaggleDR_test':
        labels = 'data/kaggle_dr/retinopathy_solution_wh.csv'

        ds = KaggleDR(path_data='data/kaggle_dr/test_JF_512',
                      filename_targets=labels,
                      preprocessing=KaggleDR.jf_trafo)
        df = pd.read_csv(labels)
        width = df.width.values.astype(theano.config.floatX)
        height = df.height.values.astype(theano.config.floatX)
    elif dataset == 'KaggleDR_train':
        labels = 'data/kaggle_dr/trainLabels_wh.csv'

        ds = KaggleDR(path_data='data/kaggle_dr/train_JF_512',
                      filename_targets=labels,
                      preprocessing=KaggleDR.jf_trafo)
        df = pd.read_csv(labels)
        width = df.width.values.astype(theano.config.floatX)
        height = df.height.values.astype(theano.config.floatX)
    else:
        print('Unknown dataset, aborting.')
        return

    det_out = np.zeros((ds.n_samples, 5), dtype=np.float32)
    stoch_out = np.zeros((ds.n_samples, 5, mc_samples), dtype=np.float32)

    idx = 0
    progbar = Progbar(ds.n_samples)
    for X, _ in ds.iterate_minibatches(np.arange(ds.n_samples),
                                       batch_size=batch_size,
                                       shuffle=False):

        n_s = X.shape[0]
        img_dim = JFnet.get_img_dim(width[idx:idx + n_s],
                                    height[idx:idx + n_s])
        det_out[idx:idx + n_s] = jfmodel.predict(X, img_dim)
        stoch_out[idx:idx + n_s] = jfmodel.mc_samples(X, img_dim,
                                                      T=mc_samples)
        idx += n_s
        progbar.add(n_s)

    det_y_pred = np.argmax(det_out, axis=1)
    det_acc = np.mean(np.equal(det_y_pred, ds.y))
    det_kappa = quadratic_weighted_kappa(det_y_pred, ds.y, 5)

    results = {'det_out': det_out,
               'stoch_out': stoch_out,
               'det_kappa': det_kappa,
               'det_acc': det_acc}

    if out_file == '{mc_samples}_mc_{dataset}_JFnet.pkl':
        out_file = out_file.format(mc_samples=mc_samples, dataset=dataset)
    with open(out_file, 'wb') as h:
        pickle.dump(results, h)

if __name__ == '__main__':
   import sys
   sys.path.append('.')    
   main()
