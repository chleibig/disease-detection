import pytest
import numpy
import theano


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


class TestJFnet:
    def test_output_jfnet(self):
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

        assert numpy.allclose(probas, probas_pred)


class TestBatchFreezableDropoutLayer:
    @pytest.fixture(params=[(100, 100), (None, 100)])
    def input_layer(self, request):
        from lasagne.layers.input import InputLayer
        return InputLayer(request.param)

    @pytest.fixture
    def layer(self, input_layer):
        from models import BatchFreezableDropoutLayer
        return BatchFreezableDropoutLayer(input_layer)

    @pytest.fixture
    def layer_no_rescale(self, input_layer):
        from models import BatchFreezableDropoutLayer
        return BatchFreezableDropoutLayer(input_layer, rescale=False)

    @pytest.fixture
    def layer_p_02(self, input_layer):
        from models import BatchFreezableDropoutLayer
        return BatchFreezableDropoutLayer(input_layer, p=0.2)

    def test_batch_freeze_true(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input, batch_freeze=True)
        result_eval = result.eval()
        assert numpy.array_equal(result_eval[0, :], result_eval[1, :])
        assert numpy.linalg.matrix_rank(result_eval) == 1

    def test_batch_freeze_false(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input, batch_freeze=False)
        result_eval = result.eval()
        assert not numpy.array_equal(result_eval[0, :], result_eval[1, :])
        assert numpy.linalg.matrix_rank(result_eval) == 100

    # Tests from here on check that by default a BatchFreezableDropoutLayer
    # behaves the same way as a DropoutLayer (source: https://github.com/
    # Lasagne/Lasagne/blob/8d57668f606bc86625bd06de75807f21643130d2/lasagne/
    # tests/layers/test_noise.py)

    def test_get_output_for_non_deterministic(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input)
        result_eval = result.eval()
        assert 0.9 < result_eval.mean() < 1.1
        assert (numpy.unique(result_eval) == [0., 2.]).all()

    def test_get_output_for_deterministic(self, layer):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer.get_output_for(input, deterministic=True)
        result_eval = result.eval()
        assert (result_eval == input.get_value()).all()

    def test_get_output_for_no_rescale(self, layer_no_rescale):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer_no_rescale.get_output_for(input)
        result_eval = result.eval()
        assert 0.4 < result_eval.mean() < 0.6
        assert (numpy.unique(result_eval) == [0., 1.]).all()

    def test_get_output_for_no_rescale_dtype(self, layer_no_rescale):
        input = theano.shared(numpy.ones((100, 100), dtype=numpy.int32))
        result = layer_no_rescale.get_output_for(input)
        assert result.dtype == input.dtype

    def test_get_output_for_p_02(self, layer_p_02):
        input = theano.shared(numpy.ones((100, 100)))
        result = layer_p_02.get_output_for(input)
        result_eval = result.eval()
        assert 0.9 < result_eval.mean() < 1.1
        assert (numpy.round(numpy.unique(result_eval), 2) == [0., 1.25]).all()

    def test_specified_rng(self, input_layer):
        from models import BatchFreezableDropoutLayer
        from lasagne.random import get_rng, set_rng
        from numpy.random import RandomState
        input = theano.shared(numpy.ones((100, 100)))
        seed = 123456789
        rng = get_rng()

        set_rng(RandomState(seed))
        result = BatchFreezableDropoutLayer(input_layer).get_output_for(input)
        result_eval1 = result.eval()

        set_rng(RandomState(seed))
        result = BatchFreezableDropoutLayer(input_layer).get_output_for(input)
        result_eval2 = result.eval()

        set_rng(rng)  # reset to original RNG for other tests
        assert numpy.allclose(result_eval1, result_eval2)


class TestModel:

    @pytest.fixture
    def net(self):
        from collections import OrderedDict
        from lasagne.layers.input import InputLayer
        from models import BatchFreezableDropoutLayer
        net = OrderedDict()
        net[0] = InputLayer((None, 100))
        net[1] = BatchFreezableDropoutLayer(net[0])
        net[2] = BatchFreezableDropoutLayer(net[1])
        return net

    @pytest.fixture
    def model(self, net):
        from models import Model
        model = Model(net)
        model.inputs['X'] = net[0].input_var
        return model

    def test_ensemble_prediction(self, model):
        x = numpy.ones((1, 100)).astype('float32')
        X = numpy.concatenate((x, x, x, x))

        pred071 = model.ensemble_prediction(x, networks=[0, 7, 1])
        pred7 = model.ensemble_prediction(x, networks=[7])
        pred7_seed_change = model.ensemble_prediction(x, networks=[7], seed=42)
        batch_pred = model.ensemble_prediction(X, networks=[0, 7, 1])

        assert pred071.shape == (1, 100, 3)
        assert pred7.shape == (1, 100, 1)
        assert batch_pred.shape == (4, 100, 3)
        assert numpy.allclose(pred071[0, :, 1], pred7[0, :, 0])
        assert not numpy.allclose(pred7, pred7_seed_change)
        assert numpy.allclose(pred071[0, :, :], batch_pred[2, :, :])
