import scipy.ndimage as nd
import scipy.ndimage.measurements as meas
from scipy.ndimage.morphology import binary_dilation as binary_dil
import numpy as np
import glob
import os
from PIL import Image
import nibabel as nib
from evaluation_comparison.pairwise_measures_GARS import RegionProperties, \
    PairwiseMeasures
import pandas as pd
import argparse
import sys
FORBIDDEN = ['MOUSE6_HA'] # To Adapt for to skip problematic folders


def create_connective_support(connection=1, dim=3):
    init = np.ones([3]*dim)
    results = np.zeros([3] * dim)
    centre = [1]*dim
    idx = np.asarray(np.where(init > 0)).T
    diff_to_centre = idx - np.tile(centre, [idx.shape[0], 1])
    sum_diff_to_centre = np.sum(np.abs(diff_to_centre), axis=1)
    idx_chosen = np.asarray(np.where(sum_diff_to_centre <= connection)).T
    np.put(results, np.squeeze(idx_chosen)[:], 1)
    # print(np.sum(results))
    return results


def create_threshsize_connected_components(seg, connection=3, thresh=5):
    connection_shape = create_connective_support(connection)
    label, nf = meas.label(seg, connection_shape)
    seg_new = np.zeros_like(label)
    for l in range(1, nf+1):
        nb_vox = np.asarray(np.where(label==l)).shape[1]
        seg_new = np.where(label==l, seg, seg_new)
        if nb_vox < thresh:
            label = np.where(label==l, np.zeros_like(label), label)
            seg_new = np.where(label == l, np.zeros_like(label), seg_new)
    return label, seg_new, nf


def study_component(label, red_filled, red, green, pixdim=[0.38, 0.38,
                                                           1.750], min_size=30):
    '''
    Performs the analysis of a given label
    Args:
        label: label number
        red_filled: filled connected component
        red: red image
        green: green image
        pixdim: pixel dimension
        min_size: minimum size with which to consider a label

    Returns:

    '''
    seg_label = np.where(red_filled == label, np.ones_like(red), np.zeros_like(
        red))
    red_label = seg_label * red
    greenred_label = seg_label * green
    greenred_min = create_threshsize_connected_components(
        greenred_label, 1, min_size)[1]
    greenred_label = red_label / np.maximum(red_label, 1) * greenred_label
    print(np.sum(red_label), np.sum(greenred_min), np.sum(greenred_label))

    results_red = RegionProperties(red_label, pixdim=pixdim)
    results_red.fill_value()
    results_greenred = RegionProperties(greenred_label, pixdim=pixdim)
    results_greenred.fill_value()
    results_greenred_min = RegionProperties(greenred_min, pixdim=pixdim)
    results_greenred_min.fill_value()
    comp_greenred = PairwiseMeasures(red_label, greenred_label, pixdim=pixdim)
    comp_greenred.fill_value()
    comp_greenred_min = PairwiseMeasures(red_label, greenred_min,
                                         pixdim=pixdim)
    comp_greenred_min.fill_value()
    return results_red.m_dict_result, results_greenred.m_dict_result, \
        results_greenred_min.m_dict_result,\
        comp_greenred.m_dict_result, comp_greenred_min.m_dict_result


def append_keys(dictionary, appending):
    for k in dictionary.keys():
        new_key = k + appending
        dictionary[new_key]  = dictionary.pop(k)
    return


def study_subject(label, red, green, subject,pixdim=None):
    '''
    Performs the per label analysis for a given subject given the RED and
    green images
    Args:
        label: connected component labelling (of the red image)
        red: red semgnetation
        green: green image
        subject: subject name
        pixdim: array with pixel dimensions

    Returns:

    '''
    if pixdim is None:
        pixdim = [0.415, 0.415, 1.750]
        pixdim = [0.38, 0.38, 1.750]
    res_red = []
    res_greenred = []
    res_greenred_min = []
    comp_greenred = []
    comp_greenred_min = []

    for l in range(1, np.max(label)+1):
        indices = np.asarray(np.where(label == l)).T
        if indices.shape[0] * np.prod(pixdim) > 50:
            r_red, r_greenred, r_greenred_min, c_greenred, c_greenred_min \
                = study_component(l, label, red,
                                                                green)
            r_red['alabel'] = l
            r_red['aid'] = subject
            append_keys(r_red, 'red')
            res_red.append(r_red)
            append_keys(r_greenred, 'gr')
            res_greenred.append(r_greenred)
            append_keys(r_greenred_min, 'grmin')
            res_greenred_min.append(r_greenred_min)
            append_keys(c_greenred, 'gr')
            comp_greenred.append(c_greenred)
            append_keys(c_greenred_min, 'grmin')
            comp_greenred_min.append(c_greenred_min)

    return pd.DataFrame(res_red), pd.DataFrame(res_greenred), \
        pd.DataFrame(res_greenred_min), pd.DataFrame(
        comp_greenred), pd.DataFrame(comp_greenred_min)


def process_subject(path_jpeg, path_save, subject_name, thresh=64, pixdim=None):
    '''
    Function to process a single subject
    Args:
        path_jpeg: path to the jpeg images
        path_save: path where to save the csv file
        subject_name: Name of the subject
        thresh: Threshold for the minimal intensity to consider
        pixdim: array with the pixel dimension

    Returns:

    '''
    affine = np.eye(4)
    list_red = glob.glob(os.path.join(path_jpeg,'*red*.jpg'))
    list_green = glob.glob(os.path.join(path_jpeg,'*green*.jpg'))
    if len(list_red) == 0:
        list_red = glob.glob(os.path.join(path_jpeg, '*RED*.jpg'))
        list_green = glob.glob(os.path.join(path_jpeg, '*GREEN*.jpg'))
    array_red = []
    array_green = []
    if len(list_red) > 0:
        for r in list_red:
            jpgfile = Image.open(r)
            array_img = np.reshape(np.asarray(list(jpgfile.getdata()))[:, 0],
                [jpgfile.width, jpgfile.height,1])
            array_red.append(array_img)

        for g in list_green:
            jpgfile = Image.open(g)
            array_img = np.reshape(np.asarray(list(jpgfile.getdata()))[:, 1],
                [jpgfile.width, jpgfile.height, 1])
            array_green.append(array_img)

        img_red = np.concatenate(array_red, 2)
        img_red = np.where(img_red > thresh, 1.0*img_red/256.0, np.zeros_like(
            img_red))
        img_red_filled = nd.binary_fill_holes(img_red)
        for z in range(0, img_red_filled.shape[2]):
            img_red_filled[...,z] = binary_dil(img_red_filled[...,z])

        label_filled, seg_filled, numb_red = \
            create_threshsize_connected_components(img_red_filled)
        nii_label = nib.Nifti1Image(label_filled, affine)
        nib.save(nii_label, os.path.join(path_save,
                                         'LabelsFilled'+subject_name+'.nii.gz'))

        img_green = np.concatenate(array_green, 2)
        img_green = np.where(img_green > thresh, 1.0*img_green/256.0,
                             np.zeros_like(
            img_red))

        res_red, res_greenred, res_greenred_min, comp_greenred, \
        comp_greenred_min\
            = study_subject(label_filled,img_red,img_green,subject_name,
                            pixdim=pixdim)

        final = pd.concat([res_red, res_greenred, res_greenred_min,
                           comp_greenred, comp_greenred_min], axis=1)
        final.to_csv(os.path.join(path_save,
                                  'ExtractedTableFinFilled_'+subject_name+'.csv'),
                                   float_format='%.3f',)

        nii_red = nib.Nifti1Image(np.concatenate(array_red, 2), affine)
        nii_green = nib.Nifti1Image(np.concatenate(array_green, 2), affine)
        nib.save(nii_red, os.path.join(path_save,
                                       'Red_'+subject_name+'.nii.gz'))
        nib.save(nii_green, os.path.join(path_save,
                                         'Green_'+subject_name+'.nii.gz'))


def main(argv):
    thresh = 64
    pixdim = [0.415, 0.415, 1.750]
    pixdim = [0.38, 0.38, 1.750]

    parser = argparse.ArgumentParser(description='Transform GIF parcellation '
                                                 'into FS parcellation.')

    parser.add_argument('-p', dest='path', metavar='input_path',
                                type=str, required=True,
                                help='path where to find the images')
    parser.add_argument('-t', dest='threshold', metavar='seg threshold',
                        type=float, default=64, help='minimum value to '
                                                     'consider a voxel as '
                                                     'positive')
    parser.add_argument('-dx', type=float, dest='pixdim', default=0.48, \
                                                             help='pixel ' \
                                                              'dimension in plane')
    parser.add_argument('-dz', type=float, dest='pixdim_z', help='pixel ' \
                                                              'dimension out of '
                                                     'plane', default=1.750)


    try :
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))

    except argparse.ArgumentTypeError:
        print('BrainHearts.py -f <filename_database> -g <grouping> -d '
              '<dependent variable> -i <independent variables>')
        print('The list of independent variables must always start with the '
              'Age')
        sys.exit(2)

    pixdim = [args.pixdim, args.pixdim, args.pixdim_z]
    thresh = args.threshold
    path_name = glob.glob(args.path)
    for p in path_name:
        dirname = os.path.dirname(p)
        s = os.path.basename(p)
        if s not in FORBIDDEN:
            pj = os.path.join(dirname, s, 'JPEG')
            ps = os.path.join(dirname, s)
            if not os.path.exists(os.path.join(dirname, s,
                                  'ExtractedTableFinFilled_'+s+'.csv')):
                try:
                    print("Attempting subject %s" % s)
                    process_subject(pj, ps, s, thresh=thresh, pixdim=pixdim)
                    print(s)
                except ValueError:
                    raise

if __name__ == "__main__":
   main(sys.argv[1:])
