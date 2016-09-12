# This script prepares the tiny imagenet data from
# https://tiny-imagenet.herokuapp.com/

path=data/tiny_imagenet
mkdir $path
wget cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d $path
mkdir $path/raw

echo 'Flattening directory tree...'
for image in $(find $path -name *.JPEG)
  do mv $image $path/raw/
done

rm -rf $path/tiny-imagenet-200
python scripts/convert_JF.py --directory=$path/raw --convert_directory=$path/JF_BG_512 --crop_size=512 --n_proc=$(nproc) --enhance_contrast=True --ignore_grayscale=True
