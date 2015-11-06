import warnings
import pickle
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import set_all_param_values
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except ImportError:
    warnings.warn("cuDNN not available, using theano's conv2d instead.")
    from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax


def vgg19(batch_size=None, input_var=None, filename=None):
    """Setup network structure for VGG19 and optionally load pretrained
    weights

    Parameters
    ----------
    batch_size : optional[int]
        if None, not known at compile time
    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.
    filename : Optional[str]
        if filename is not None, weights are loaded from filename

    Returns
    -------
    dict
        one lasagne layer per key

    Notes
    -----
        Reference: Simonyan & Zisserman, 2015: "Very Deep Convolutional
        Networks for Large-Scale Image Recognition"
        This function is based on:
            https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
        License: non-commercial use only
        Download pretrained weights from:
        https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl

    """

    net = {}
    net['input'] = InputLayer((batch_size, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    if filename is not None:
        load_weights(net['prob'], filename)

    return net


def vgg19_fc7_to_prob(batch_size=None, input_var=None, filename=None,
                      n_classes=5):
    """Setup network structure for VGG19 layers fc7 and softmax output

       Input is supposed to be feature activations of fc6 layer from VGG19

    Parameters
    ----------
    batch_size : optional[int]
        if None, not known at compile time
    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.
    filename : Optional[str]
        if filename is not None, weights are loaded from filename
    n_classes : Optional[int]
        default 5 for transfer learning on Kaggle DR data

    Returns
    -------
    dict
        one lasagne layer per key

    """

    net = {}
    net['input'] = InputLayer((batch_size, 4096), input_var=input_var)
    net['fc7'] = DenseLayer(net['input'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=n_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    if filename is not None:
        load_weights(net['prob'], filename)

    return net


def vgg19_fc8_to_prob(batch_size=None, input_var=None, filename=None,
                      n_classes=5):
    """Setup network structure for retraining last layer of VGG19

       Input is supposed to be feature activations of fc7 layer from VGG19
       The network architecture is equivalent to logistic regression

    Parameters
    ----------
    batch_size : optional[int]
        if None, not known at compile time
    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.
    filename : Optional[str]
        if filename is not None, weights are loaded from filename
    n_classes : Optional[int]
        default 5 for transfer learning on Kaggle DR data

    Returns
    -------
    dict
        one lasagne layer per key

    """

    net = {}
    net['input'] = InputLayer((batch_size, 4096), input_var=input_var)
    net['fc8'] = DenseLayer(net['input'], num_units=n_classes,
                            nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    if filename is not None:
        load_weights(net['prob'], filename)

    return net


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
        with np.load('model.npz') as f:
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
