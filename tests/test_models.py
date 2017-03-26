import numpy.testing as nt


# def test_jfnet():
#     """Verify correct reimplementation of Jeffrey de Fauw's net

#        This test had passed formerly but is now disabled for convenience
#        because it requires the repository

#        https://github.com/JeffreyDF/kaggle_diabetic_retinopathy

#        and for this in turn an older version of lasagne
#        (e.g. commit cf1a23c21666fc0225a05d284134b255e3613335)

#     """
#     import cPickle as pickle
#     import sys
#     import lasagne.layers as ll
#     from models import JFnet

#     repo_jeffrey_df = '/path/to/JeffreyDF/kaggle_diabetic_retinopathy'
#     sys.path.append(repo_jeffrey_df)

#     filename_ref = 'models/jeffrey_df/2015_07_17_123003.pkl'
#     filename = 'models/jeffrey_df/2015_07_17_123003_PARAMSDUMP.pkl'

#     model_ref = pickle.load(open(filename_ref, 'r'))
#     l_out_ref = model_ref['l_out']

#     network = JFnet.build_model(filename=filename, batch_size=64)
#     l_out = network['31']

#     # check weights and biases for equality
#     for pvs, pvs_ref in zip(ll.get_all_param_values(l_out),
#                             ll.get_all_param_values(l_out_ref)):
#         nt.assert_array_equal(pvs, pvs_ref)

#     # check layers for equality
#     for l, l_ref in zip(ll.get_all_layers(l_out), ll.get_all_layers(
#             l_out_ref)):
#         assert l.output_shape == l_ref.output_shape


def test_output_jfnet():
    import numpy as np

    from datasets import KaggleDR
    from models import JFnet

    jfmodel = JFnet(width=512, height=512)

    probas = np.array([[9.38881755e-01, 5.23291342e-02, 8.59508850e-03,
                        1.34651185e-04, 5.94010562e-05],
                       [9.19074774e-01, 6.69652745e-02, 1.35666728e-02,
                        2.82015972e-04, 1.11185553e-04]], dtype=np.float32)

    kdr = KaggleDR(path_data='tests/ref_data/KDR/sample_JF_512',
                   filename_targets='tests/ref_data/KDR/sampleLabels.csv',
                   preprocessing=KaggleDR.jf_trafo)

    X, _ = next(kdr.iterate_minibatches(np.arange(kdr.n_samples),
                                        batch_size=2,
                                        shuffle=False))
    width = height = np.array(X.shape[0] * [512], dtype=np.float32)

    probas_pred = jfmodel.predict(X, JFnet.get_img_dim(width, height))

    nt.assert_array_almost_equal(probas, probas_pred)


def test_bcnn2_b69aadd():
    import numpy as np

    from datasets import KaggleDR
    import models

    det_out = np.array([[0.89167184, 0.10832818],
                        [0.92720264, 0.07279734],
                        [0.90044737, 0.09955268],
                        [0.90154523, 0.09845474],
                        [0.01999491, 0.98000503],
                        [0.44038445, 0.55961555],
                        [0.67832261, 0.32167742],
                        [0.92573458, 0.07426542],
                        [0.94570011, 0.05429989],
                        [0.95900822, 0.04099183]], dtype=np.float32)

    ds = KaggleDR(path_data='tests/ref_data/KDR/sampleTest_JF_BG_512',
                  filename_targets='tests/ref_data/KDR/sampleLabelsTest.csv',
                  preprocessing=KaggleDR.standard_normalize)
    pred_det = np.zeros((ds.n_samples, 2), dtype=np.float32)

    model = models.JFnetMono(p_conv=0.2, last_layer='17',
                             weights='models/weights_bcnn2_b69aadd.npz')

    idx = 0
    for X, _ in ds.iterate_minibatches(np.arange(ds.n_samples),
                                       batch_size=2,
                                       shuffle=False):
        n_s = X.shape[0]
        pred_det[idx:idx + n_s] = model.predict(X)
        idx += n_s

    nt.assert_array_almost_equal(det_out, pred_det, decimal=4)
