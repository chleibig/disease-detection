""" Extract a set of labels from OR data

    Some data cleaning is necessary as we seem to have just a partial export of
    their database. Some observations and informations:

    - Comparison of images under data/ and fotos.nombre_foto (ignoring
      subfolder numbers):
      3779 images under data that are not listed under fotos.nombre_foto
      2663 images listed under fotos.nombre_foto that are not on disk

    - Comparison of id from list_of_cases sheet (labels) with caso_id from
      fotos sheet (presumably the foreign key to labels):
      we have 37 more labels than listed images and in turn one listed image
      for which we don't have a label.

    Summing up, we have overlapping but not identical sets for the three
    sources of information that we need to combine (1.images on disk; 2.link
    between images on disk and cases; 3.link between cases and labels)

    -> We have to work on the intersection of all three sets

    author: christian.leibig@uni-tuebingen.de
"""

import os
import pandas as pd
import numpy as np

data_path = '/media/cl/My Book/OR/data'
sheet_path = '/media/cl/My Book/OR/tables/export_tables'
labels_file = os.path.join(sheet_path,'OR_diseased_labels.csv')

# labels
with open(os.path.join(sheet_path,'list_of_cases.xls')) as h:
    cases_to_labels = pd.read_excel(h)

# link between labels and filenames
with open(os.path.join(sheet_path,'fotos.xlsx')) as h:
    images_to_cases = pd.read_excel(h)

# image files available on disk
images_on_disk = pd.DataFrame(
    [(fn, dir) for dir in os.listdir(data_path)
                 for fn in os.listdir(os.path.join(data_path, dir, 'linked'))],
     columns=['nombre_foto', 'centro_id'])

# select only those image to case links for which we have images
images_to_cases['centro_id'] = images_to_cases['centro_id'].astype(str)
images_to_cases = pd.merge(images_to_cases, images_on_disk,
                           how='inner', on=['nombre_foto', 'centro_id'],
                           copy=False)

cases_to_labels.rename(columns={'id':'caso_id'}, inplace=True)
# link image with label information:
il = pd.merge(images_to_cases, cases_to_labels,
                            how='inner', on='caso_id')

# As a first step create healthy vs. diseased labels, especially as
# different diseases might be simultaneously present, i.e. one might need to
# assign each sample to multiple classes. Whether this is necessary can
# be empirically verified by analysing the respective columns.

# Christian Wojek queried a single set of labels based on columns
# signos_rpd_*, signos_dmae_* and otras_alteraciones.
# "otras_alteraciones" seems to contain lots of different formulations,
# for the moment we take "valoracion" instead.

VALORACION_HEALTHY = {u'Normal', u'normal'}
SIGNOS_RPD_HEALTHY = {u'Nein', u'No', u'Not', u'nein'}
SIGNOS_DMAE_HEALTHY = {u'Nein', u'No', u'No signos de DMAE', u'Not', u'nein'}

healthy = ((il.valoracion.isin(VALORACION_HEALTHY).values) &
           (il.signos_rpd_od.isin(SIGNOS_RPD_HEALTHY).values) &
           (il.signos_rpd_oi.isin(SIGNOS_RPD_HEALTHY).values) &
           (il.signos_dmae_od.isin(SIGNOS_DMAE_HEALTHY).values) &
           (il.signos_dmae_oi.isin(SIGNOS_DMAE_HEALTHY).values))

diseased = np.zeros(len(il,), dtype=np.int)
diseased[~healthy] = 1
il['diseased'] = diseased

EVALUABLE = {u'Auswertbar', u'Completamente evaluable', u'Evaluable for',
             u'auswertbar'}
# Currently we do not have a direct link between labels and images. Hence
# we apply some heuristics in order to avoid label noise
n_images_for_case = il.caso_id.value_counts()
CASES_WITH_TWO_IMAGES = n_images_for_case.index[n_images_for_case.values == 2]
# exclude caso_ids for which non FUNDUS images were observed
UNWANTED_CASES = {4545, 4546, 11023, 14167, 22670}


row_selection = (
                  (il.centro_id_x.astype(int).values ==
                   il.centro_id_y.astype(int).values) &
                  (il.calidad_retinografia_od.isin(EVALUABLE).values) &
                  (il.calidad_retinografia_oi.isin(EVALUABLE).values) &
                  (il.signos_rpd_od.values ==  il.signos_rpd_oi.values) &
                  (il.signos_dmae_od.values == il.signos_dmae_oi.values) &
                  (il.caso_id.isin(CASES_WITH_TWO_IMAGES).values) &
                  (~il.caso_id.isin(UNWANTED_CASES).values)
                )

col_selection = ['nombre_foto', 'centro_id_x', 'caso_id', 'diseased']

il.ix[row_selection, col_selection].to_csv(labels_file, index=False,
                                           header = ['filename', 'centre_id',
                                                     'case_id', 'diseased'])