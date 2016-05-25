import os
import pandas as pd
from PIL import Image
import click


@click.command()
@click.option('--fname_labels',
              default='data/kaggle_dr/retinopathy_solution.csv',
              show_default=True,
              help="Filename labels.")
@click.option('--path', default='data/kaggle_dr/test',
              show_default=True,
              help="Path to full sized images.")
def main(fname_labels, path):
    """Get width and height from images and write new labels file with
       this additional information
    """

    df = pd.read_csv(fname_labels)

    df['width'] = 0
    df['height'] = 0

    with click.progressbar(df.iterrows(), length=len(df),
                           label='Appending image dimensions') as row_iterator:
        for idx, row in row_iterator:
            fname = os.path.join(path, row['image'] + '.jpeg')
            im = Image.open(fname, mode='r')
            df.ix[idx, 'width'], df.ix[idx, 'height'] = im.size

    df.to_csv(fname_labels.replace('.csv', '_wh.csv'), index=False)

if __name__ == '__main__':
    main()
