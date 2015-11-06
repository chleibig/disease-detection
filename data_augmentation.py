"""This code is largely based on team o_O's data augmentation routines which
were obtained from
https://github.com/sveitser/kaggle_diabetic/blob/master/data.py
which in turn is based on
https://github.com/benanne/kaggle-ndsb/blob/master/data.py

"""
from __future__ import division, print_function
import os
import time
import click
import numpy as np
from PIL import Image
import theano
import skimage.transform
from skimage.transform._warps_cy import _warp_fast
import pandas as pd
from datasets import KaggleDR

# for color augmentation, computed with make_pca.py
U = np.array([[-0.56543481, 0.71983482, 0.40240142],
              [-0.5989477, -0.02304967, -0.80036049],
              [-0.56694071, -0.6935729, 0.44423429]],
             dtype=theano.config.floatX)
EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=theano.config.floatX)

NO_AUGMENTATION_PARAMS = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def fast_warp(img, tf, output_shape, mode='constant', order=0):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params
    t_img = np.zeros((img.shape[0],) + output_shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, output_shape=output_shape,
                              mode=mode, order=order)
    return t_img


def build_centering_transform(image_shape, target_shape):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(
        translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the
    center of the image. Use these transforms to center and uncenter the
    image around such a transform.
    """

    # need to swap rows and cols here apparently! confusing!
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5
    tform_uncenter = skimage.transform.SimilarityTransform(
        translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(
        translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0,
                                 translation=(0, 0), flip=False):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(
        scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation),
        shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def random_perturbation_transform(zoom_range, rotation_range, shear_range,
                                  translation_range, do_flip=True,
                                  allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0)  # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1]
    # instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear,
                                        translation, flip)


def perturb(img, augmentation_params, target_shape, rng=np.random):
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = random_perturbation_transform(rng=rng,
                                                  **augmentation_params)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform_augment = tform_uncenter + tform_augment + tform_center
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape,
                     mode='constant')


# for test-time augmentation
def perturb_fixed(img, tform_augment, target_shape=(50, 50)):
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    # shift to center, augment, shift back (for the rotation/shearing)
    tform_augment = tform_uncenter + tform_augment + tform_center
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape, mode='constant')


def augment_color(img, sigma=0.1, color_vec=None):

    if color_vec is None:
        if not sigma > 0.0:
            color_vec = np.zeros(3, dtype=theano.config.floatX)
        else:
            color_vec = np.random.normal(0.0, sigma, 3)

    alpha = color_vec.astype(theano.config.floatX) * EV
    noise = np.dot(U, alpha.T)
    return img + noise[:, np.newaxis, np.newaxis]


def load_image(filename):
    """Load image

    Parameters
    ----------
    filename : string

    Returns
    -------
    image : numpy array, shape = (n_colors, n_rows, n_columns), dtype =
                                                           theano.config.floatX
    """
    return np.array(Image.open(filename), dtype=theano.config.floatX)\
        .transpose(2, 0, 1)


def augment(img, w, h, aug_params=NO_AUGMENTATION_PARAMS,
                 transform=None, sigma=0.0, color_vec=None):
    """Augment image with output shape (w, h).

    Default arguments return non augmented image of shape (w, h).
    To apply a fixed transform (color augmentation) specify transform
    (color_vec).
    To generate a random augmentation specify aug_params and sigma.

    Parameters
    ----------
    image : numpy array, shape = (n_colors, n_rows, n_columns), dtype =
                                                           theano.config.floatX
        source image

    Returns
    -------
    image : numpy array, shape = (n_colors, n_rows, n_columns), dtype =
                                                           theano.config.floatX
        transformed image

    Note
    ----
    Kaggle DR data is occasionally flipped along the horizontal axis due to
    differences in acquisition systems. Therefore, the preferred axis to
    flip along for augmentation is as well the horizontal one. Flipping is
    implemented via shearing by 180 degrees followed by a
    rotation by 180 degrees, requiring the input image to be transposed.
    Therefore this function temporarily transposes row and column directions.

    """

    img = img.transpose(0, 2, 1)

    if transform is None:
        img = perturb(img, augmentation_params=aug_params, target_shape=(w, h))
    else:
        img = perturb_fixed(img, tform_augment=transform, target_shape=(w, h))

    img = img.transpose(0, 2, 1)

    img = KaggleDR.standard_normalize(img)

    img = augment_color(img, sigma=sigma, color_vec=color_vec)
    return img


def augment_labels(filename_labels_org, filename_labels_aug):
    """Augment label.csv file for data augmentation

    Parameters
    ----------
    filename_labels_org : string
        file with original labels
    filename_labels_aug : string
        file to save original plus augmented labels to

    Returns
    -------
    labels_aug : pandas DataFrame with columns 'image' and 'level'

    """

    labels_org = pd.read_csv(filename_labels_org, dtype={'level': np.int32})
    y = labels_org['level']

    classes = np.unique(y)
    n_samples = len(y)
    priors = np.array([np.count_nonzero(y == c_k) for c_k in classes])\
             /n_samples
    balance_weights = 1/priors
    n_aug = np.ceil(balance_weights - 1).astype(np.int32)
    n_wanted = np.multiply(balance_weights, priors * n_samples).astype(
        np.int32)
    n_total = (priors * n_samples).astype(np.int32)
    labels_aug = labels_org.copy()
    #TODO: Preallocate size of final labels_aug instead of appending
    for i, k in enumerate(y):
        if n_total[k] < n_wanted[k]:
            n_aug_k = min(n_aug[k], n_wanted[k] - n_total[k])
            org_file = labels_org['image'][i]
            aug_files = [org_file+'_aug_'+str(j) for j in range(n_aug_k)]
            aug_labels = [k] * n_aug_k
            to_append = pd.DataFrame(zip(aug_files, aug_labels),
                                     columns=['image', 'level'])
            labels_aug = labels_aug.append(to_append, ignore_index=True)
            n_total[k] += n_aug_k

    labels_aug.to_csv(filename_labels_aug)
    return labels_aug


@click.command()
@click.option('--source_dir', default=None, show_default=True,
              help="Directory with original images.")
@click.option('--filename_targets', default=None, show_default=True,
              help="Absolute filename of trainLabels.csv")
@click.option('--extension', default='jpeg', show_default=True,
              help="Extension of source image files")
@click.option('--outfile', default='images.npy', show_default=True,
              help="Numpy memory mapped array for original and augmented "
                   "images.")
def main(source_dir, filename_targets, extension, outfile):
    """ Augment data according to team_o_O """
    cnf = {
        'augmentation_params': {'zoom_range': (1 / 1.15, 1.15),
                                'rotation_range': (0, 360),
                                'shear_range': (0, 0),
                                'translation_range': (-20, 20),
                                'do_flip': True,
                                'allow_stretch': True},
        'sigma': 0.1,
        'w': 224,
        'h': 224
    }

    print("Augmenting labels...")
    start_time = time.time()
    labels_aug = augment_labels(filename_targets,
                                filename_targets.split('.')[0] + '_aug.csv')
    print("Augmentation of labels took",
          np.round((time.time() - start_time), 3), "sec.")

    print("Augmenting images...")
    start_time = time.time()
    fp = np.memmap(outfile, dtype=theano.config.floatX, mode='w+',
                   shape=(len(labels_aug), 3, cnf['h'], cnf['w']))

    for i, fn in enumerate(labels_aug['image']):
        if 'aug' in fn:
            fn = os.path.join(source_dir, fn[:fn.find('_aug')]+'.'+extension)
            img = load_image(fn)
            fp[i] = augment(img, cnf['w'], cnf['h'],
                            aug_params=cnf['augmentation_params'],
                            sigma=cnf['sigma'])
        else:
            fn = os.path.join(source_dir, fn+'.'+extension)
            img = load_image(fn)
            fp[i] = KaggleDR.standard_normalize(img)

    print("Augmentation of images took",
          np.round((time.time() - start_time), 3), "sec.")

if __name__ == '__main__':
    main()
