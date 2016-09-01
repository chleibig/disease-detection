# This script prepares the messidor data and should be run
# after downloading all zip files from 
# http://www.adcis.net/en/Download-Third-Party/Messidor.htmldownload-en.php
# into data/messidor

path=data/messidor

for zipped in $path/Base*.zip
  do unzip $zipped
done

mkdir $path/raw
mv $path/*.tif $path/raw/
rm Base*.zip

python -c "from datasets import Messidor; Messidor.prepare_labels()"

rm $path/*.xls

python scripts/convert_JF.py --directory=$path/raw --convert_directory=$path/JF_BG_512 --crop_size=512 --n_proc=$(nproc) --enhance_contrast=True

python scripts/append_img_dim.py -l $path/messidor.csv -p $path/raw -e tif