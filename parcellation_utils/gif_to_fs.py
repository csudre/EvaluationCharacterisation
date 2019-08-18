import nibabel as nib
import numpy as np
import csv
import os
import argparse
import sys
import pandas as pd


def map_gif_to_fs(parcellation_file):
    df = pd.read_csv(os.path.join(os.path.split(os.path.abspath(
        __file__))[0], 'FS_GIF.csv'))
    parc = nib.load(parcellation_file)
    data = parc.get_data()
    new_mapping = np.zeros_like(data)
    for v in df.FS:
        df_temp = df.loc[df.FS == v]
        if isinstance(df_temp['GIF'].data.obj[0], str):
            val_gif = np.fromstring(df_temp['GIF'].data.obj[0], dtype=int, sep=' ')
            for g in val_gif:
                new_mapping = np.where(data == g, v * np.ones_like(data),
                                   new_mapping)

    final_nii = nib.Nifti1Image(new_mapping, parc.affine)
    return final_nii


def relabel_over_mask(parcellation, mask, label):
    parc_nii = nib.load(parcellation)
    mask_nii = nib.load(mask)
    label_nii = nib.load(label)
    new_parc = parc_nii.get_data()
    mask_data = mask_nii.get_data()
    label_data = label_nii.get_data()
    new_parc = np.where(mask_data > 0, label_data, new_parc)
    new_nii = nib.Nifti1Image(new_parc, parc_nii.affine)
    return new_nii




