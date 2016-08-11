import warnings
import pickle
import numpy as np
import lasagne

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
except ImportError:
    warnings.warn("cuDNN not available, using theano's conv2d instead.")
    from lasagne.layers import Conv2DLayer as ConvLayer
    from lasagne.layers import MaxPool2DLayer

from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import set_all_param_values

from lasagne.nonlinearities import softmax, LeakyRectify


class JFnet(object):

    @staticmethod
    def build_model(input_var=None, width=512, height=512, filename=None,
                    n_classes=5, batch_size=None, p_conv=0.0):
        """Setup network structure for the original formulation of JeffreyDF's
           network and optionally load pretrained weights

        Parameters
        ----------
        input_var : Theano symbolic variable or `None` (default: `None`)
            A variable representing a network input. If it is not provided, a
            variable will be created.
        width : Optional[int]
            image width
        height : Optional[int]
            image height
        filename : Optional[str]
            if filename is not None, weights are loaded from filename
        n_classes : Optional[int]
            default 5 for transfer learning on Kaggle DR data
        batch_size : should only be set if all batches have the same size!
        p_conv: dropout applied to conv. layers, by default turned off (0.0)

        Returns
        -------
        dict
            one lasagne layer per key

        Notes
        -----
            Reference: Jeffrey De Fauw, 2015:
            http://jeffreydf.github.io/diabetic-retinopathy-detection/

            Download pretrained weights from:
            https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/
            master/dumps/2015_07_17_123003_PARAMSDUMP.pkl

           original net has leaky rectifier units

        """

        net = {}

        net['0'] = InputLayer((batch_size, 3, width, height),
                              input_var=input_var, name='images')
        net['1'] = ConvLayer(net['0'], 32, 7, stride=(2, 2), pad='same',
                             untie_biases=True,
                             nonlinearity=LeakyRectify(leakiness=0.5),
                             W=lasagne.init.Orthogonal(1.0),
                             b=lasagne.init.Constant(0.1))
        net['1d'] = DropoutLayer(net['1'], p=p_conv)
        net['2'] = MaxPool2DLayer(net['1d'], 3, stride=(2, 2))
        net['3'] = ConvLayer(net['2'], 32, 3, stride=(1, 1), pad='same',
                             untie_biases=True,
                             nonlinearity=LeakyRectify(leakiness=0.5),
                             W=lasagne.init.Orthogonal(1.0),
                             b=lasagne.init.Constant(0.1))
        net['3d'] = DropoutLayer(net['3'], p=p_conv)
        net['4'] = ConvLayer(net['3d'], 32, 3, stride=(1, 1), pad='same',
                             untie_biases=True,
                             nonlinearity=LeakyRectify(leakiness=0.5),
                             W=lasagne.init.Orthogonal(1.0),
                             b=lasagne.init.Constant(0.1))
        net['4d'] = DropoutLayer(net['4'], p=p_conv)
        net['5'] = MaxPool2DLayer(net['4d'], 3, stride=(2, 2))
        net['6'] = ConvLayer(net['5'], 64, 3, stride=(1, 1), pad='same',
                             untie_biases=True,
                             nonlinearity=LeakyRectify(leakiness=0.5),
                             W=lasagne.init.Orthogonal(1.0),
                             b=lasagne.init.Constant(0.1))
        net['6d'] = DropoutLayer(net['6'], p=p_conv)
        net['7'] = ConvLayer(net['6d'], 64, 3, stride=(1, 1), pad='same',
                             untie_biases=True,
                             nonlinearity=LeakyRectify(leakiness=0.5),
                             W=lasagne.init.Orthogonal(1.0),
                             b=lasagne.init.Constant(0.1))
        net['7d'] = DropoutLayer(net['7'], p=p_conv)
        net['8'] = MaxPool2DLayer(net['7d'], 3, stride=(2, 2))
        net['9'] = ConvLayer(net['8'], 128, 3, stride=(1, 1), pad='same',
                             untie_biases=True,
                             nonlinearity=LeakyRectify(leakiness=0.5),
                             W=lasagne.init.Orthogonal(1.0),
                             b=lasagne.init.Constant(0.1))
        net['9d'] = DropoutLayer(net['9'], p=p_conv)
        net['10'] = ConvLayer(net['9d'], 128, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['10d'] = DropoutLayer(net['10'], p=p_conv)
        net['11'] = ConvLayer(net['10d'], 128, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['11d'] = DropoutLayer(net['11'], p=p_conv)
        net['12'] = ConvLayer(net['11d'], 128, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['12d'] = DropoutLayer(net['12'], p=p_conv)
        net['13'] = MaxPool2DLayer(net['12d'], 3, stride=(2, 2))
        net['14'] = ConvLayer(net['13'], 256, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['14d'] = DropoutLayer(net['14'], p=p_conv)
        net['15'] = ConvLayer(net['14d'], 256, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['15d'] = DropoutLayer(net['15'], p=p_conv)
        net['16'] = ConvLayer(net['15'], 256, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['16d'] = DropoutLayer(net['16'], p=p_conv)
        net['17'] = ConvLayer(net['16d'], 256, 3, stride=(1, 1), pad='same',
                              untie_biases=True,
                              nonlinearity=LeakyRectify(leakiness=0.5),
                              W=lasagne.init.Orthogonal(1.0),
                              b=lasagne.init.Constant(0.1))
        net['17d'] = DropoutLayer(net['17'], p=p_conv)
        net['18'] = MaxPool2DLayer(net['17d'], 3, stride=(2, 2),
                                   name='coarse_last_pool')
        net['19'] = DropoutLayer(net['18'], p=0.5)
        net['20'] = DenseLayer(net['19'], num_units=1024, nonlinearity=None,
                               W=lasagne.init.Orthogonal(1.0),
                               b=lasagne.init.Constant(0.1),
                               name='first_fc_0')
        net['21'] = FeaturePoolLayer(net['20'], 2)
        net['22'] = InputLayer((batch_size, 2), name='imgdim')
        net['23'] = ConcatLayer([net['21'], net['22']])
        # Combine representations of both eyes
        net['24'] = ReshapeLayer(net['23'],
                                 (-1, net['23'].output_shape[1] * 2))
        net['25'] = DropoutLayer(net['24'], p=0.5)
        net['26'] = DenseLayer(net['25'], num_units=1024, nonlinearity=None,
                               W=lasagne.init.Orthogonal(1.0),
                               b=lasagne.init.Constant(0.1),
                               name='combine_repr_fc')
        net['27'] = FeaturePoolLayer(net['26'], 2)
        net['28'] = DropoutLayer(net['27'], p=0.5)
        net['29'] = DenseLayer(net['28'],

                               num_units=n_classes * 2,
                               nonlinearity=None,
                               W=lasagne.init.Orthogonal(1.0),
                               b=lasagne.init.Constant(0.1))
        # Reshape back to the number of desired classes
        net['30'] = ReshapeLayer(net['29'], (-1, n_classes))
        net['31'] = NonlinearityLayer(net['30'], nonlinearity=softmax)

        if filename is not None:
            with open(filename, 'r') as f:
                weights = pickle.load(f)
            set_all_param_values(net['31'], weights)

        return net

    @staticmethod
    def get_img_dim(width, height, idx, n_samples):
        """Second input to JFnet consumes image dimensions

        division by 700 according to https://github.com/JeffreyDF/
        kaggle_diabetic_retinopathy/blob/
        43e7f51d5f3b2e240516678894409332bb3767a8/generators.py::lines 41-42
        """
        img_dim = np.vstack((width[idx:idx + n_samples],
                             height[idx:idx + n_samples])).T / 700.
        return img_dim


def load_weights(layer, filename):
    """
    Load network weights from either a pickle or a numpy file and set
    the parameters of all layers below layer (including the layer itself)
    to the given values.

    Parameters
    ----------
    layer : Layer
        The :class:`Layer` instance for which to set all parameter values
    filename : str with ending .pkl or .npz

    """

    if filename.endswith('.npz'):
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        set_all_param_values(layer, param_values)
        return

    if filename.endswith('.pkl'):
        with open(filename) as handle:
            model = pickle.load(handle)
        set_all_param_values(layer, model['param values'])
        return

    raise NotImplementedError('Format of {filename} not known'.format(
        filename=filename))


def save_weights(layer, filename):
    """
    Save network weights of all layers below layer (including the layer
    itself).

    Parameters
    ----------
    layer : Layer or list
        The :class:`Layer` instance for which to gather all parameter values,
        or a list of :class:`Layer` instances.
    filename : str with ending .npz

    """

    if filename.endswith('.npz'):
        np.savez_compressed(filename,
                            *lasagne.layers.get_all_param_values(layer))
        return

    raise NotImplementedError('Format indicated by ending of {filename} not'
                              'implemented'.format(filename=filename))
