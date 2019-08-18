import nibabel as nib
import numpy as np
import os
from scipy import ndimage
from .bullseye_plotting import (create_bullseye_plot, prepare_data_bullseye,
                                read_ls_create_agglo, prepare_data_fromagglo,
                                LABELS_LR, FULL_LABELS)
import matplotlib.pyplot as plt
import sys
import getopt
import argparse
from nifty_utils.file_utils import split_filename


def creation_ls(lobe_file, layer_file, lesion_file):
    '''
    Create local summary
    :param lobe_file: nifti files with lobar separation
    :param layer_file: nifti file with layer separation
    :param lesion_file: nifti file with lesion segmentation
    :return:
    '''
    lobe_nii = nib.load(lobe_file)
    layer_nii = nib.load(layer_file)
    lesion_nii = nib.load(lesion_file)
    lobe = lobe_nii.get_data()
    layer = layer_nii.get_data()
    lesion = lesion_nii.get_data()
    vol_vox = np.prod(lobe_nii.header.get_zooms())
    max_lobe = np.max(lobe)
    max_layer = np.max(layer)
    vol_prob = []
    vol_bin = []
    vol_reg = []
    connect = []
    layer = np.where(layer == max_layer, (max_layer-1)*np.ones_like(layer),
                     layer)
    [connected, num_connect] = ndimage.measurements.label(lesion)
    for lobe_index in range(0, max_lobe):
        for layer_index in range(1, max_layer):
            region_lobe = np.where(lobe == lobe_index+1, np.ones_like(lobe),
                                   np.zeros_like(lobe))
            region_layer = np.where(layer == layer_index, np.ones_like(layer),
                                    np.zeros_like(layer))
            region = np.multiply(region_lobe, region_layer)
            lesion_region = np.multiply(region, lesion)
            vol_prob.append(np.sum(lesion_region)*vol_vox)
            vol_bin.append(np.where(lesion_region > 0)[0].shape[0]*vol_vox)
            values = np.unique(connected*lesion_region)
            connect.append(len(values)-1)
            vol_reg.append(np.sum(region)*vol_vox)
    vol_prob.append(np.sum(lesion)*vol_vox)
    vol_bin.append(np.where(lesion > 0)[0].shape[0]*vol_vox)
    vol_reg.append(np.where(lobe*layer > 0)[0].shape[0]*vol_vox)
    connect.append(num_connect)
    return vol_prob, vol_bin, vol_reg, connect


def write_ls(vol_prob, vol_bin, vol_reg, connect, filewrite):
    '''
    Write the information of local summary to filewrite
    :param vol_prob: Probabilistic lesion volumes
    :param vol_bin: Binary lesion volumes
    :param vol_reg: Regional volumes
    :param connect: Information on connected components per region
    :param filewrite: File to write to
    :return:
    '''
    with open(filewrite, 'w') as out_stream:
        for (vprob, vbin, vreg, con) in zip(vol_prob, vol_bin, vol_reg,
                                            connect):
            out_stream.write(str(vprob)+" " + str(vbin)+' '+str(vreg)+' ' +
                             str(con)+'\n')


def bullseyes_from_nii(lobe_file, layer_file, lesion_file, filewrite):
    '''
    Create bullseye plot from lobar, layer and lesion nii images
    :param lobe_file: Lobar separation
    :param layer_file: Layer separation
    :param lesion_file: Lesion segmentation
    :param filewrite: File on which to write to
    :return:
    '''
    vol_prob, vol_bin, vol_reg, connect = creation_ls(lobe_file, layer_file,
                                                      lesion_file)
    write_ls(vol_prob, vol_bin, vol_reg, connect, filewrite)
    path = os.path.split(lesion_file)[0]
    name = os.path.split(lesion_file)[1]
    v_perc, v_dist = prepare_data_bullseye(filewrite)
    create_bullseye_plot(v_perc, 'YlOrRd', 0, 0.25)
    plt.show()
    plt.savefig(os.path.join(path, 'BEPerc_' + name.rstrip('.nii.gz') + '.png'))
    create_bullseye_plot(v_dist, 'seismic', 0, 0.1)
    plt.show()






