
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage.morphology as morph
#from matplotlib import pyplot as plt
from os import path
import sys
import argparse
#from skimage.morphology import convex_hull_image as chi
sys.path.append(path.abspath(
    '/Users/csudre/PycharmProjects/EvaluationCharacterisation'
                            ))

from evaluation_comparison.region_properties import RegionProperties, \
    LIST_SHAPE, LIST_HARALICK, LIST_HIST
import os


# In[2]:





# In[3]:


def get_labels_periv(connected_nii, parc_nii):
    parc = parc_nii.get_data()
    parc_ventr = np.where(np.logical_and(parc<54, parc>49), np.ones_like(parc), np.zeros_like(parc))
    parc_dil = morph.binary_dilation(parc_ventr)
    connect = connected_nii.get_data()
    overlap = np.where(parc_dil>0, connect, np.zeros_like(connect))
    return np.unique(overlap)[1:]


# In[4]:



# In[5]:


def create_subdivided_fromlist(connected_nii, list_labels):
    connect = connected_nii.get_data()
    selected = np.zeros_like(connect)
    for lab in list_labels:
        selected += np.where(connect==lab, np.ones_like(connect)*lab, np.zeros_like(connect))
    return selected
    




def main(argv):
    # path_data = '/Users/csudre/Documents/UK_BBK/TempWork'
    # data_label = pd.read_csv(
    #     os.path.join(path_data, 'ParsedLesion_4059650.csv'))
    # connected_nii = nib.load(
    #     os.path.join(path_data, 'Connect_WS3WT3WC1Lesion_4059650_corr.nii.gz'))
    # parc_nii = nib.load(
    #     os.path.join(path_data, 'GIF_Parcellation_4059650.nii.gz'))
    # mahal_nii = nib.load(
    #     os.path.join(path_data, 'LesionMahal_T1FLAIR_BiASM_4059650_TA.nii.gz'))
    # lobes_nii = nib.load(os.path.join(path_data, 'Lobes_4059650.nii.gz'))

    parser = argparse.ArgumentParser(description='Create csv file with'
                                                 ' region properties')
    parser.add_argument('-connect', dest='connected', metavar='connected '
                                                                'pattern',
                        type=str, required=True,
                        help='RegExp pattern for the connected files')
    parser.add_argument('-parsed', dest='parsed', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-path', dest='path', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-name', dest='name', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-mahal', dest='mahal', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-parc', dest='parc', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-lobes', dest='lobes', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-type', dest='type', action='store',
                        default='paired', type=str, help='indicates if mask '
                                                         'and image should be paired')
    parser.add_argument('-a', dest='analysis', default='binary', choices=[
        'binary', 'label', 'cc'], help='indicates how the mask should be '
                                      'treated: binary, per label or per '
                                       'connected component', action='store',
                        type =str)
    parser.add_argument('-t', dest='threshold', action='store', default=0.5,
                        type=float, help='threshold to apply to get a binary '
                                         'mask')
    parser.add_argument('-neigh', dest='neighborhood', default=1,
                        choices=[1, 2, 3], action='store', type=int,
                        help='type of neighborhood applied when creating the '
                             'connected component structure')
    parser.add_argument('-meas', dest='measures', default=['simple'],
                        nargs='+', help='list of measures to be extrated')
    parser.add_argument('-mul', dest='mul', action='store', type=float,
                        default=None, help='multiplicative value for the '
                        'intensities')
    parser.add_argument('-trans', dest='trans', action='store', type=float,
                        default=None, help='offset value for the intensities')


    try:
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compute_ROI_statistics.py -i <input_image_pattern> -m '
              '<mask_image_pattern> -t <threshold> -mul <analysis_type> '
              '-trans <offset>   ')
        sys.exit(2)

    path_data = args.path
    connected_nii = nib.load(args.connected)
    parc_nii = nib.load(args.parc)
    lobes_nii = nib.load(args.lobes)
    mahal_nii = nib.load(args.mahal)


    list_peri = get_labels_periv(connected_nii, parc_nii)
    cc_sel = create_subdivided_fromlist(connected_nii, list_peri)


    lobes = lobes_nii.get_data()
    periv_FL = np.where(lobes == 1, cc_sel, np.zeros_like(cc_sel))
    periv_FR = np.where(lobes == 2, cc_sel, np.zeros_like(cc_sel))
    periv_POL = np.where(np.logical_or(np.logical_or(lobes == 3, lobes == 5),
                                       lobes==7), cc_sel,
                         np.zeros_like(cc_sel))
    periv_POR = np.where(np.logical_or(np.logical_or(lobes == 4, lobes == 6),
                                       lobes==8),
                         cc_sel,
                         np.zeros_like(cc_sel))
    list_periv = [np.expand_dims(periv_FL, -1), np.expand_dims(periv_FR, -1),
                  np.expand_dims(periv_POL, -1), np.expand_dims(periv_POR, -1)]
    stacked_periv = np.concatenate(list_periv, -1)
    new_nii = nib.Nifti1Image(stacked_periv, connected_nii.affine)
    nib.save(new_nii, os.path.join(path_data,
                                   'PerivSplit_'+args.name+'.nii.gz'))

    rp_FL = RegionProperties(periv_FL, mahal_nii.get_data(),
                             LIST_SHAPE + LIST_HIST + LIST_HARALICK,
                             pixdim=mahal_nii.header.get_zooms()[0:3])
    rp_FL.fill_value()
    rp_FR = RegionProperties(periv_FR, mahal_nii.get_data(),
                             LIST_SHAPE + LIST_HIST + LIST_HARALICK,
                             pixdim=mahal_nii.header.get_zooms()[0:3])
    rp_FR.fill_value()

    rp_POL = RegionProperties(periv_POL, mahal_nii.get_data(),
                              LIST_SHAPE + LIST_HIST + LIST_HARALICK,
                              pixdim=mahal_nii.header.get_zooms()[0:3])
    rp_POL.fill_value()

    rp_POR = RegionProperties(periv_POR, mahal_nii.get_data(),
                              LIST_SHAPE + LIST_HIST + LIST_HARALICK,
                              pixdim=mahal_nii.header.get_zooms()[0:3])
    rp_POR.fill_value()
    rp_FL.header_str()
    pd_FL = pd.DataFrame.from_dict([rp_FL.m_dict_result])
    pd_FR = pd.DataFrame.from_dict([rp_FR.m_dict_result])
    pd_POL = pd.DataFrame.from_dict([rp_POL.m_dict_result])
    pd_POR = pd.DataFrame.from_dict([rp_POR.m_dict_result])
    csv_FL = os.path.join(args.path, 'CSVFL_'+args.name+'.csv')
    csv_FR = os.path.join(args.path, 'CSVFR_' + args.name + '.csv')
    csv_POL = os.path.join(args.path, 'CSVPOL_' + args.name + '.csv')
    csv_POR = os.path.join(args.path, 'CSVPOR_' + args.name + '.csv')
    pd_FL.to_csv(csv_FL)
    pd_FR.to_csv(csv_FR)
    pd_POL.to_csv(csv_POL)
    pd_POR.to_csv(csv_POR)

if __name__ == "__main__":
    main(sys.argv[1:])