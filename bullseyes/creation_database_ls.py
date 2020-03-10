import numpy as np
import sys
import getopt
import glob
import pandas as pd
import os
import argparse
import nibabel as nib
from scipy import ndimage
from bullseyes.bullseye_plotting import (create_bullseye_plot,
                                     prepare_data_bullseye,
                                read_ls_create_agglo, prepare_data_fromagglo,
                                LABELS_LR, FULL_LABELS)
import matplotlib.pyplot as plt



TYPES = ['Les', 'Reg', 'Freq', 'Dist']
LOBES = ['F', 'P', 'O', 'T', 'BG', 'IT']
TERR = ['IT','PCA','MCA','ACA']
SIDES = ['L', 'R']
COMBINED = ['BG', 'IT']


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





def create_name_save(list_format):
    list_elements = []
    common_path = os.path.split(os.path.commonprefix(list_format))[0]
    print(common_path)
    list_common = common_path.split(os.sep)
    for l in list_format:
        split_string = l.lstrip(common_path).split(os.sep)
        for s in split_string:
            if s not in list_common and s not in list_elements:
                list_elements.append(s.replace("*", '_'))
    return common_path, '_'.join(list_elements)


def create_header_foragglo(numb_layers=4, side=None, lobes=None):
    if side is None:
        side = SIDES
    if lobes is None:
        lobes = LOBES
    header = ['ID']
    for type_value in TYPES:
        head_full = [type_value+'Tot']
        head_layers = []
        head_lobes = []
        head_lobeslayers_sides = []
        head_lobeslayers_full = []
        head_lobes_side = []
        for layer_value in range(0, numb_layers):
            head_layers.append(type_value + str(layer_value + 1))
        for lobe_value in lobes:
            if lobe_value not in COMBINED:
                head_lobes.append(type_value+lobe_value)
                for layer_value in range(0, numb_layers):
                    head_lobeslayers_sides.append(type_value+lobe_value +
                                                  str(layer_value+1))
            for side_value in side:
                if lobe_value not in COMBINED:
                    head_lobes_side.append(type_value+lobe_value+side_value)
                    for layer_value in range(0, numb_layers):
                        head_lobeslayers_full.append(type_value+lobe_value +
                                                     side_value +
                                                     str(layer_value+1))

        combined = ""
        if 'BG' in lobes and 'IT' in lobes:
            for lobe_value in COMBINED:
                if lobe_value in lobes:
                    head_lobes_side.append(type_value+lobe_value)
                    combined += lobe_value
                    for layer_value in range(0, numb_layers):
                        head_lobeslayers_full.append(type_value + lobe_value + str(
                        layer_value+1))
        head_lobes.append(type_value+combined)
        for layer_value in range(0, numb_layers):
            head_lobeslayers_sides.append(type_value + combined +
                                          str(layer_value + 1))
        header += head_full + head_lobeslayers_full + head_layers + \
            head_lobes_side + head_lobeslayers_sides + head_lobes
    return header


def create_header_foragglo_corr(numb_layers=4, side=SIDES, lobes=LOBES):
    if side is None:
        side = SIDES
    if lobes is None:
        lobes = LOBES
    header = []
    for type_value in TYPES:
        head_full = [type_value+'Tot']
        head_layers = []
        head_lobes = []
        head_lobeslayers_sides = []
        head_lobeslayers_full = []
        head_lobes_side = []
        for layer_value in range(0, numb_layers):
            head_layers.append(type_value + str(layer_value + 1))
        for lobe_value in lobes:
            if lobe_value not in COMBINED:
                head_lobes.append(type_value + lobe_value)
                for layer_value in range(0, numb_layers):
                    head_lobeslayers_sides.append(type_value + lobe_value +
                                                  str(layer_value + 1))
            for side_value in side:
                if lobe_value not in COMBINED:
                    head_lobes_side.append(type_value + lobe_value + side_value)
                    for layer_value in range(0, numb_layers):
                        head_lobeslayers_full.append(type_value + lobe_value +
                                                     side_value +
                                                     str(layer_value + 1))

        combined = ""
        for lobe_value in COMBINED:
            head_lobes_side.append(type_value + lobe_value)
            combined += lobe_value
            for layer_value in range(0, numb_layers):
                head_lobeslayers_full.append(type_value + lobe_value
                                             + str(layer_value + 1))
        head_lobes.append(type_value + combined)
        for layer_value in range(0, numb_layers):
            head_lobeslayers_sides.append(type_value + combined +
                                          str(layer_value + 1))
        header += head_full + head_lobeslayers_full + head_layers + \
            head_lobes_side + head_lobeslayers_sides + head_lobes
    return header




