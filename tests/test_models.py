import numpy.testing as nt


def test_jeffrey_df():
    """Verify correct reimplementation of Jeffrey de Fauw's net"""
    import cPickle as pickle
    import sys
    import lasagne.layers as ll
    import models

    # loading the model_ref further down requires access to layers module from
    # jeffrey de fauw's repo
    repo_jeffrey_df = '/home/cl/Downloads/kdr_solutions/JeffreyDF/' \
                      'kaggle_diabetic_retinopathy'
    sys.path.append(repo_jeffrey_df)

    filename_ref = 'models/jeffrey_df/2015_07_17_123003.pkl'
    filename = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'

    model_ref = pickle.load(open(filename_ref, 'r'))
    l_out_ref = model_ref['l_out']

    network = models.jeffrey_df(filename=filename, batch_size=64)
    l_out = network['31']

    # check weights and biases for equality
    for pvs, pvs_ref in zip(ll.get_all_param_values(l_out),
                            ll.get_all_param_values(l_out_ref)):
        nt.assert_array_equal(pvs, pvs_ref)

    # check layers for equality
    for l, l_ref in zip(ll.get_all_layers(l_out), ll.get_all_layers(
            l_out_ref)):
        assert l.output_shape == l_ref.output_shape


def test_output_jeffrey_df():
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne

    from datasets import KaggleDR
    import models

    weights = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'
    network = models.jeffrey_df(width=512, height=512, filename=weights,
                                untie_biases=True)

    expected = np.array([[9.38881755e-01, 5.23291342e-02, 8.59508850e-03,
                          1.34651185e-04, 5.94010562e-05],
                         [9.19074774e-01, 6.69652745e-02, 1.35666728e-02,
                          2.82015972e-04, 1.11185553e-04]], dtype=np.float32)

    X = T.tensor4('inputs')
    network['0'].input_var = X
    img_dim = T.tensor4('img_dim')
    network['22'].input_var = img_dim
    prob = lasagne.layers.get_output(network['31'], deterministic=True)

    forward_pass = theano.function([X, img_dim], prob)

    kdr = KaggleDR(path_data='tests/ref_data/KDR/sample_JF_512',
                   filename_targets='tests/ref_data/KDR/sampleLabels.csv',
                   preprocessing=KaggleDR.jf_trafo)

    inputs, targets = next(kdr.iterate_minibatches(np.arange(kdr.n_samples),
                                                   batch_size=2, shuffle=False))
    n_s = len(targets)
    _img_dim = np.concatenate(
        (np.full((n_s, 1, 1, 1), inputs.shape[2], dtype=np.float32),
         np.full((n_s, 1, 1, 1), inputs.shape[3], dtype=np.float32)),
        axis=1)/700.  # jeffrey does that under load_image_and_process?!
    output =  forward_pass(inputs, _img_dim)
    output = np.reshape(output, (2, 5))

    nt.assert_array_almost_equal(expected, output)
