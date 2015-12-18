"""Crop and resize images according to Jeffrey de Fauw, save with desired
extension.

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


def convert(fname, crop_size):
    """Refactored from JF's generators.load_image_and_process"""
    im = Image.open(fname, mode='r')

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

    return converted


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return left, upper, right, lower


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('jpeg', extension).replace(directory, 
                                                    convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory, 
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname) 


def save(img, fname):
    img.save(fname, quality=97)


@click.command()
@click.option('--directory', default='data/train', show_default=True,
              help="Directory with original images.")
@click.option('--convert_directory', default='data/train_res',
              show_default=True,
              help="Where to save converted images.")
@click.option('--crop_size', default=256, show_default=True,
              help="Size of converted images.")
@click.option('--extension', default='tiff', show_default=True,
              help="Filetype of converted images.")
@click.option('--n_proc', default=2, show_default=True,
              help="Number of processes for parallelization.")
def main(directory, convert_directory, crop_size, extension, n_proc):
    """Image preprocessing according to Jeffrey de Fauw:
       Crop and resize images, save with desired extension.
    """

    try:
        os.mkdir(convert_directory)
    except OSError:
        pass

    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory)
                 for f in fn if f.endswith('jpeg') or f.endswith('tiff')] 
    filenames = sorted(filenames)

    print("Resizing images in {} to {}, this takes a while."
          "".format(directory, convert_directory))

    n = len(filenames)
    # process in batches, sometimes weird things happen with Pool on my machine
    batchsize = 500
    batches = n // batchsize + 1
    pool = Pool(n_proc)

    args = []

    for f in filenames:
        args.append((convert, (directory, convert_directory, f, crop_size, 
                               extension)))

    for i in range(batches):
        print("batch {:>2} / {}".format(i + 1, batches))
        pool.map(process, args[i * batchsize: (i + 1) * batchsize])

    pool.close()

    print('done')

if __name__ == '__main__':
    main()
