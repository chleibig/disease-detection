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








