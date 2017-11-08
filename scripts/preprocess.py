"""Image preprocessing: configurable combination of cropping, resizing and
colour contrast enhancement.

this code is based on:
https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py
and:
https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master
/generators.py: load_image_and_process

christian.leibig@uni-tuebingen.de, 2015

"""


from __future__ import division, print_function
import os
from multiprocessing.pool import Pool

import click
import numpy as np
from PIL import Image
import cv2


white_list_extensions = ['jpg', 'jpeg', 'JPEG', 'tif']


def convert(fname, crop_size, enhance_contrast=False, ignore_grayscale=False):
    """Refactored from JF's generators.load_image_and_process"""
    im = Image.open(fname, mode='r')

    if not ignore_grayscale:
        assert len(np.shape(im)) == 3, "Shape of image {} unexpected, " \
            "maybe it's grayscale".format(fname)
    elif len(np.shape(im)) == 2:
        print(fname, 'seems to be grayscale, ignored.')
        return None

    w, h = im.size

    if w / float(h) >= 1.3:
        cols_thres = np.where(
            np.max(
                np.max(
                    np.asarray(im),
                    axis=2),
                axis=0) > 35)[0]

        # Extra cond compared to orig crop.
        if len(cols_thres) > crop_size // 2:
            min_x, max_x = cols_thres[0], cols_thres[-1]
        else:
            min_x, max_x = 0, -1

        converted = im.crop((min_x, 0, max_x, h))

    else:  # no crop
        converted = im

    # Resize without preserving aspect ratio:
    converted = converted.resize((crop_size, crop_size),
                                 resample=Image.BILINEAR)

    if enhance_contrast:
        im_ce_as_array = contrast_enhance(np.asarray(converted),
                                          radius=crop_size // 2)
        converted = Image.fromarray(im_ce_as_array.astype(np.uint8))

    return converted


def contrast_enhance(im, radius):
    """Subtract local average color and map local average to 50% gray

    Parameters
    ==========

    im: array of shape (height, width, 3)
    radius: int
        for square images a good choice is size/2

    Returns
    =======

    im_ce: contrast enhanced image as array of shape (height, width, 3)

    Reference
    =========

    B. Graham, "Kaggle diabetic retinopathy detection competition report",
        University of Warwick, Tech. Rep., 2015

    https://github.com/btgraham/SparseConvNet/blob/kaggle_Diabetic_Retinopathy_competition/Data/kaggleDiabeticRetinopathy/preprocessImages.py

    """

    radius = int(radius)
    b = np.zeros(im.shape)
    cv2.circle(b, (radius, radius), int(radius * 0.9),
               (1, 1, 1), -1, 8, 0)
    im_ce = cv2.addWeighted(im, 4,
                            cv2.GaussianBlur(im, (0, 0), radius / 30),
                            -4, 128) * b + 128 * (1 - b)
    return im_ce


def get_convert_fname(fname, extension, directory, convert_directory):
    source_extension = fname.split('.')[-1]
    return fname.replace(source_extension,
                         extension).replace(directory,
                                            convert_directory)


def create_dirs(paths):
    for p in paths:
        try:
            os.makedirs(p)
        except OSError:
            pass


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, \
        extension, enhance_contrast, ignore_grayscale = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size, enhance_contrast, ignore_grayscale)
        if img is not None:
            save(img, convert_fname)


def save(img, fname):
    img.save(fname, quality=97)


@click.command()
@click.option('--source_dir', default='data/train', show_default=True,
              help="Directory with original images.")
@click.option('--target_dir', default='data/train_res',
              show_default=True,
              help="Where to save converted images.")
@click.option('--crop_size', default=512, show_default=True,
              help="Size of converted images.")
@click.option('--extension', default='jpeg', show_default=True,
              help="Filetype of converted images.")
@click.option('--n_proc', default=1, show_default=True,
              help="Number of processes for parallelization.")
@click.option('--enhance_contrast', is_flag=True,
              default=True, show_default=True,
              help="Whether to use Benjamin Graham's contrast enhancement.")
@click.option('--ignore_grayscale', is_flag=True,
              default=False, show_default=True,
              help="Whether to ignore grayscale images.")
def main(source_dir, target_dir, crop_size,
         extension, n_proc, enhance_contrast, ignore_grayscale):
    """Image conversion: crop, resize, (enhance colour contrast) and save."""

    try:
        os.mkdir(target_dir)
    except OSError:
        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(source_dir)
                 for f in fn if f.split('.')[-1] in white_list_extensions]

    assert filenames, "No valid filenames."

    print("Resizing images in {} to {}, this takes a while."
          "".format(source_dir, target_dir))

    n = len(filenames)
    batchsize = 500
    batches = n // batchsize + 1
    pool = Pool(n_proc)

    args = []

    for f in filenames:
        args.append((convert, (source_dir, target_dir, f, crop_size,
                               extension, enhance_contrast, ignore_grayscale)))

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')


if __name__ == '__main__':
    main()
