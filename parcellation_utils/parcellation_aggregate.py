import nibabel as nib
import numpy as np
import os

import pandas as pd


def create_hemisphere(filename):
    """
    Create two hemispheres
    :param filename: parcellation data
    :return: hemispheric segmentation
    """
    left_array = (31, 33, 38, 40, 42, 44, 49, 51, 53, 55,
                  57, 59, 61, 63, 65, 67, 76, 89, 90,
                  91, 92, 93, 94, 97, 102, 104, 106, 108, 110,
                  114, 116, 118, 120, 122, 124, 126, 130, 134, 136, 138,
                  140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 162, 164,
                  166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188,
                  192, 194, 196, 198, 200, 202, 204, 206, 208)
    right_array = (24, 32, 37, 39, 41, 43, 48, 50, 52, 54, 56, 58, 60, 62,
                   64, 66, 77, 81, 82, 83, 84, 85, 86, 96, 101, 103, 105, 107,
                   109, 113, 115, 117, 119, 121, 123, 125, 129, 133, 135, 137,
                   139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 161,
                   163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183,
                   185, 187, 191, 193, 195, 197, 199, 201, 203, 205, 207)
    img = nib.load(filename)
    img_data = img.get_data()
    right_hemi = np.zeros_like(img_data)
    left_hemi = np.zeros_like(img_data)
    for v_r in right_array:
        right_hemi[img_data == v_r] = 1
        right_nii = nib.Nifti1Image(right_hemi, img.affine)
    for v_l in left_array:
        left_hemi[img_data == v_l] = 1
        left_nii = nib.Nifti1Image(left_hemi, img.affine)
    return right_nii, left_nii


def create_bg(filename):
    """
    Create the DGM aggregation from the parcellation file
    :param filename: Parcellation file to be used for the segmentation
    :return: Segmented DGM as nifti image
    """

    bg_array = (24,	31,	32,	33,	37,	38,	56,	57,	58,	59,	60,	61,	76,	77)
    img = nib.load(filename)
    img_data = img.get_data()
    bg_seg = np.zeros_like(img_data)
    for value_r in bg_array:
        bg_seg[img_data == value_r] = 1
    bg_nii = nib.Nifti1Image(bg_seg, img.affine)
    return bg_nii


def prepare_use_gif_hierarchy():
    """
    Prepare the dictionary and dataframe to use the GIF hierarchy
    :return: gif hierarchy and dictionary of structures and their level
    """
    gif_h = pd.read_csv(os.path.join(os.path.split(os.path.abspath(
        __file__))[0], 'KeysHierarchy_ordered.csv'))

    choices_level_1 = list(set(gif_h['Label_1']))
    choices_level_2 = list(set(gif_h['Label_2']))
    choices_level_3 = list(set(gif_h['Label_3']))
    choices_level_34 = list(set(gif_h['Label_34']))
    choices_level_4 = list(set(gif_h['Label_4']))
    choices_level_5 = list(set(gif_h['Label_5']))
    choices_level_6 = list(set(gif_h['Label_Full']))

    dict_1 = {c: 'Label_1' for c in choices_level_1}
    dict_2 = {c: 'Label_2' for c in choices_level_2}
    dict_3 = {c: 'Label_3' for c in choices_level_3}
    dict_34 = {c: 'Label_34' for c in choices_level_34}
    dict_4 = {c: 'Label_4' for c in choices_level_4}
    dict_5 = {c: 'Label_5' for c in choices_level_5}
    dict_6 = {c: 'Label_6' for c in choices_level_6}

    dict_total = dict_1
    dict_total.update(dict_2)
    dict_total.update(dict_3)
    dict_total.update(dict_34)
    dict_total.update(dict_4)
    dict_total.update(dict_5)
    dict_total.update(dict_6)

    return gif_h, dict_total


def create_aggregated_volume(parc_data, aggregation_name, gif_h, dict_levels):
    """
    Provide the segmentation volume (number of voxels) for a given
    aggregation name from which corresponding labels will be found
    :param parc_data: parcellation data
    :param aggregation_name: name of the aggregated structure to measure
    :param gif_h: gif hierarchy as pd dataframe
    :param dict_levels: dictionary of levels and available aggregated structures
    :return: volume
    """

    if aggregation_name not in dict_levels.keys():
        print(list_suggestion_aggregates(dict_levels, aggregation_name))
        return 0
    else:
        level = dict_levels[aggregation_name]
        labels = gif_h[gif_h[level] == aggregation_name]['Label_Full'].array
        vol_aggregated = create_volume_from_labellist(parc_data, labels)
        return vol_aggregated


def create_volume_from_labellist(parc_data, label_list):
    """
    Provide the volume of aggregated structures listed in label list
    :param parc_data: parcellation data
    :param label_list: list of labels to aggregate
    :return: number of voxels in given structure
    """
    vol = 0
    for label in label_list:
        vol += np.reshape(np.asarray(np.where(parc_data == label)),
                          [3, -1]).shape[-1]
    return vol


def create_seg_from_labellist(parc_data, label_list):
    """
    Create segmentation from list of labels
    :param parc_data: parcellation data
    :param label_list: list of labels to aggregate
    :return: segmentation
    """
    seg_results = np.zeros_like(parc_data)
    for label in label_list:
        seg_results = np.where(parc_data == label, np.ones_like(parc_data),
                               seg_results)
    return seg_results


def create_aggregation(parc_data, aggregation_name, gif_h, dict_levels):
    """
    Create the aggregation using the name and the labelling hierarchy
    :param parc_data: parcellation data
    :param aggregation_name: name of the aggregated structure
    :param gif_h: gif hierarchy as dataframe
    :param dict_levels: dictionary of possible structures
    :return: segmentation of the desired structure
    """
    if aggregation_name not in dict_levels.keys():
        print(list_suggestion_aggregates(dict_levels, aggregation_name))
        return np.zeros_like(parc_data)
    else:
        level = dict_levels[aggregation_name]
        labels = gif_h[gif_h[level] == aggregation_name]['Label_Full'].array
        seg_aggregated = create_seg_from_labellist(parc_data, labels)
        return seg_aggregated


def list_suggestion_aggregates(dict_levels, string_matched):
    """
    Provide the list of suggested aggregates with close match to initial
    suggested string
    :param dict_levels: list of structures as dictionary
    :param string_matched: string to match
    :return: list of potential structures of interest
    """
    import re
    list_matched = []
    print(dict_levels.keys())
    for keydict in dict_levels.keys():
        match = re.search(string_matched, str(keydict), re.IGNORECASE)
        if match:
            list_matched.append(keydict)
    return list_matched


def combine_seg(list_seg, combination='combined_binary'):
    """
    Provide a combination of multiple segmented structures as either
    independent segmentation, a binary combination, a labelled segmentation
    or 4D aggregation
    :param list_seg: list of segmentation arrays
    :param combination: type of combination to be chosen among 'separated',
    'combined_binary' 'label_3d' and 'split_4d' default is 'combined_binary'
    :return: the aggregation in a list
    """
    if combination == 'separated':
        return list_seg
    if combination == 'label_3d':
        segmentation_final = np.zeros_like(list_seg[0])
        range_labels = np.arange(0, len(list_seg)) + 1
        for (seg, ind) in zip(list_seg, range_labels):
            segmentation_final += np.where(seg == 1, ind * np.ones_like(seg),
                                           segmentation_final)
        return [segmentation_final]
    if combination == 'split_4d':
        list_new_seg = []
        for seg in list_seg:
            list_new_seg.append(np.expand_dims(seg, -1))
        segmentation_final = np.concatenate(list_new_seg, -1)
        return [segmentation_final]
    else:
        segmentation_final = np.zeros_like(list_seg[0])
        for seg in list_seg:
            segmentation_final += np.where(seg == 1, np.ones_like(seg),
                                           segmentation_final)
        return [segmentation_final]

