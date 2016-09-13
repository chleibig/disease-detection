# This script prepares the imagenet val. data from ILSVRC2012
# as obtained from http://image-net.org/download

path=data/imagenet2012_val
mkdir $path
mkdir $path/raw
tar -C $path/raw -xvf ILSVRC2012_img_val.tar

python scripts/convert_JF.py --directory=$path/raw --convert_directory=$path/JF_BG_512 --crop_size=512 --n_proc=$(nproc) --enhance_contrast --ignore_grayscale
