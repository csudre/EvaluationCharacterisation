from __future__ import absolute_import, print_function

import os.path
import glob
import nibabel as nib
import numpy as np
import getopt
import sys
import argparse
from evaluation_comparison.pairwise_measures import PairwiseMeasures
from evaluation_comparison.pairwise_measures import MorphologyOps
from nifty_utils.file_utils import (create_name_save, reorder_list_presuf)

MEASURES = (
    'ref volume', 'seg volume',
    'tp', 'fp', 'fn',
    'connected_elements', 'vol_diff',
    'outline_error', 'detection_error',
    'fpr', 'ppv', 'npv', 'sensitivity', 'specificity',
    'accuracy', 'jaccard', 'dice', 'ave_dist', 'haus_dist', 'haus_dist95'
)
MEASURES_LABELS = ('ref volume', 'seg volume', 'list_labels', 'tp', 'fp', 'fn',
                   'vol_diff', 'fpr', 'ppv', 'sensitivity', 'specificity',
                   'accuracy', 'jaccard', 'dice', 'ave_dist', 'haus_dist',
                   'com_dist', 'com_ref', 'com_seg')

# MEASURES_NEW = ('ref volume', 'seg volume', 'tp', 'fp', 'fn', 'outline_error',
#             'detection_error', 'dice')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'PairwiseMeasure'


class Parameters:
    def __init__(self, seg_format, ref_format, save_name,
                 threshold=0.5, analysis='binary', step=0.1, save_maps=True):
        self.save_name = save_name
        self.threshold = threshold
        self.ref_format = ref_format
        self.seg_format = seg_format
        self.analysis = analysis
        self.step = step
        self.save_maps = save_maps


def run_compare(param):
    # output
    list_format = [param.seg_format, param.ref_format]
    dir_init, name_save_init = create_name_save(list_format)
    out_name = '{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        param.save_name,
        param.analysis)
    iteration = 0
    while os.path.exists(os.path.join(dir_init, out_name)):
        iteration += 1
        out_name = '{}_{}_{}_{}.csv'.format(
            OUTPUT_FILE_PREFIX,
            param.save_name,
            param.analysis, str(iteration))

    print("Writing {} to {}".format(out_name, dir_init))

    # inputs
    seg_names_init = glob.glob(param.seg_format)
    ref_names_init = glob.glob(param.ref_format)
    seg_names = []
    ref_names = []
    # seg_names = util.list_files(param.seg_dir, param.ext)
    # ref_names = util.list_files(param.ref_dir, param.ext)
    ind_s, ind_r = reorder_list_presuf(seg_names_init, ref_names_init)
    print(len(ind_s))
    for i in range(0, len(ind_s)):
        if ind_s[i] > -1:
            print(i, ind_s[i])
            print(seg_names_init[i], ref_names_init[
                ind_s[i]])
            seg_names.append(seg_names_init[i])
            ref_names.append(ref_names_init[ind_s[i]])
    pair_list = list(zip(seg_names, ref_names))
    # TODO check seg_names ref_names matching
    # TODO do we evaluate all combinations?
    # import itertools
    # pair_list = list(itertools.product(seg_names, ref_names))
    print("List of references is {}".format(ref_names))
    print("List of segmentations is {}".format(seg_names))

    # prepare a header for csv
    with open(os.path.join(dir_init, out_name), 'w+') as out_stream:
        # a trivial PairwiseMeasures obj to produce header_str
        if param.analysis == 'discrete':
            m_headers = PairwiseMeasures(0, 0,
                                         measures=MEASURES_LABELS).header_str()
            out_stream.write("Name (ref), Name (seg), Label" + m_headers + '\n')
            measures_fin = MEASURES_LABELS
        else:
            m_headers = PairwiseMeasures(0, 0,
                                         measures=MEASURES).header_str()
            out_stream.write("Name (ref), Name (seg), Label" + m_headers + '\n')
            measures_fin = MEASURES

        # do the pairwise evaluations
        for i, pair_ in enumerate(pair_list):
            seg_name = pair_[0]
            ref_name = pair_[1]
            print('>>> {} of {} evaluations, comparing {} and {}.'.format(
                i + 1, len(pair_list), ref_name, seg_name))
            seg_nii = nib.load(seg_name)
            ref_nii = nib.load(ref_name)
            voxel_sizes = seg_nii.header.get_zooms()[0:3]
            seg = np.squeeze(seg_nii.get_data())
            ref = ref_nii.get_data()
            assert (np.all(seg) >= 0)
            assert (np.all(ref) >= 0)
            assert (seg.shape == ref.shape)
            # Create and save nii files of map of differences (FP FN TP OEMap
            #  DE if flag_save_map is on and binary segmentation
            flag_createlab = False
            if (param.analysis == 'discrete') and (np.max(seg) <= 1):
                flag_createlab = True
                seg = np.asarray(seg >= param.threshold)
                ref = np.asarray(ref >= param.threshold)
                blob_ref = MorphologyOps(ref, 6).foreground_component()
                ref = blob_ref[0]
                blob_seg = MorphologyOps(seg, 6).foreground_component()
                seg = blob_seg[0]
                if param.save_maps:
                    label_ref_nii = nib.Nifti1Image(ref, ref_nii.affine)
                    label_seg_nii = nib.Nifti1Image(seg, seg_nii.affine)
                    name_ref_label = os.path.join(dir_init,
                                                  'LabelsRef_'+os.path.split(
                                                      ref_name)[1])
                    name_seg_label = os.path.join(dir_init,
                                                  'LabelsSeg_'+os.path.split(
                                                      seg_name)[1])
                    nib.save(label_ref_nii, name_ref_label)
                    nib.save(label_seg_nii, name_seg_label)

                #     and (len(np.unique(seg)) > 2):
                # print('Non-integer class labels for discrete analysis')
                # print('Thresholding to binary map with threshold: {}'.format(
                #     param.threshold))
                # seg = np.asarray(seg >= param.threshold, dtype=np.int8)

            ## TODO: user specifies how to convert seg -> seg_binary
            if param.analysis == 'discrete':
                print('Discrete analysis')
                threshold_steps = np.unique(ref)
            elif param.analysis == 'prob':
                print('Probabilistic analysis')
                threshold_steps = np.arange(0, 1, param.step)
            else:
                print('Binary analysis')
                threshold_steps = [param.threshold]

            for j in threshold_steps:
                if j == 0:
                    continue
                list_labels_seg = []
                if j >= 1:
                    if not flag_createlab:  # discrete eval with same labels
                        seg_binary = np.asarray(seg == j, dtype=np.float32)
                        ref_binary = np.asarray(ref == j, dtype=np.float32)

                    else:
                        # different segmentations with connected components
                        #  (for instance lesion segmentation)
                        ref_binary = np.asarray(ref == j, dtype=np.float32)
                        seg_matched = np.multiply(ref_binary, seg)
                        list_labels_seg = np.unique(seg_matched)
                        seg_binary = np.zeros_like(ref_binary)
                        for l in list_labels_seg:
                            if l > 0:
                                seg_temp = np.asarray(seg == l)
                                seg_binary = seg_binary + seg_temp
                        print(np.sum(seg_binary))

                elif j < 1:  # prob or binary eval
                    seg_binary = np.asarray(seg >= j, dtype=np.float32)
                    ref_binary = np.asarray(ref >= 0.5, dtype=np.float32)
                    if param.save_maps and param.analysis == 'binary':
                        # Creation of the error maps per type and saving
                        temp_pe = PairwiseMeasures(seg_binary, ref_binary,
                                                   measures=(
                                                       'outline_error'),
                                                   num_neighbors=6,
                                                   pixdim=voxel_sizes)
                        tp_map, fn_map, fp_map = \
                            temp_pe.connected_errormaps()
                        intersection = np.multiply(seg_binary, ref_binary)
                        oefp_map = np.multiply(tp_map, seg_binary) - \
                            intersection
                        oefn_map = np.multiply(tp_map, ref_binary) - \
                            intersection
                        oefp_nii = nib.Nifti1Image(oefp_map, ref_nii.affine)
                        oefn_nii = nib.Nifti1Image(oefn_map, ref_nii.affine)
                        tp_nii = nib.Nifti1Image(intersection,
                                                 ref_nii.affine)
                        defp_nii = nib.Nifti1Image(fp_map, ref_nii.affine)
                        defn_nii = nib.Nifti1Image(fn_map, ref_nii.affine)
                        defn_name = os.path.join(dir_init,
                                                 param.save_name +
                                                 '_DEFN_' + os.path.split(
                                                     seg_name)[1])
                        defp_name = os.path.join(dir_init,
                                                 param.save_name +
                                                 '_DEFP_' + os.path.split(
                                                     seg_name)[1])
                        oefn_name = os.path.join(dir_init,
                                                 param.save_name +
                                                 '_OEFN_' + os.path.split(
                                                     seg_name)[1])
                        oefp_name = os.path.join(dir_init,
                                                 param.save_name +
                                                 '_OEFP_' + os.path.split(
                                                     seg_name)[1])
                        tp_name = os.path.join(dir_init,
                                               param.save_name +
                                               '_TP_' + os.path.split(
                                                   seg_name)[1])

                        nib.save(oefn_nii, oefn_name)
                        nib.save(oefp_nii, oefp_name)
                        nib.save(tp_nii, tp_name)
                        nib.save(defp_nii, defp_name)
                        nib.save(defn_nii, defn_name)
                if np.all(seg_binary == 0):
                    # Have to put default results.
                    print("Empty foreground in thresholded binary image.")
                    pe = PairwiseMeasures(seg_binary, ref_binary,
                                          measures=measures_fin,
                                          num_neighbors=6,
                                          pixdim=voxel_sizes, empty=True)
                else:
                    pe = PairwiseMeasures(seg_binary, ref_binary,
                                          measures=measures_fin,
                                          num_neighbors=6,
                                          pixdim=voxel_sizes,
                                          list_labels=list_labels_seg)
                if len(list_labels_seg) > 0 and 'list_labels' in measures_fin:
                    pe.list_labels = list_labels_seg
                fixed_fields = "{}, {}, {},".format(ref_name, seg_name, j)
                out_stream.write(fixed_fields + pe.to_string(
                    OUTPUT_FORMAT) + '\n')
                out_stream.flush()
                os.fsync(out_stream.fileno())
    out_stream.close()


def main(argv):

    parser = argparse.ArgumentParser(description='Create evaluation file when'
                                                 ' comparing two segmentations')
    parser.add_argument('-s', dest='seg_format', metavar='seg pattern',
                        type=str, required=True,
                        help='RegExp pattern for the segmentation files')
    parser.add_argument('-r', dest='ref_format', action='store',
                        default='', type=str,
                        help='RegExp pattern for the reference files')
    parser.add_argument('-t', dest='threshold', action='store', default=0.5,
                        type=float)
    parser.add_argument('-a', dest='analysis', action='store', type=str,
                        default='binary',choices=['binary','discrete','prob'])
    parser.add_argument('-save_name', dest='save_name', action='store',
                        default='', help='name to save results')
    parser.add_argument('-save_maps', dest='save_maps', action='store_true',
                        help='flag to indicate that the maps of differences '
                             'and error should be saved')
    try:
        args = parser.parse_args()
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compare_segmentation.py -s <segmentation_pattern> -r '
              '<reference_pattern> -t <threshold> -a <analysis_type> '
              '-save_name <name for saving> -save_maps  ')

        sys.exit(2)

    param = Parameters(args.seg_format, args.ref_format,
                       threshold=args.threshold,
                       save_name=args.save_name, analysis=args.analysis,
                       save_maps=args.save_maps)
    run_compare(param)


if __name__ == "__main__":
   main(sys.argv[1:])