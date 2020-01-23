from __future__ import absolute_import, print_function

import os.path
import sys
import getopt
import glob
import numpy as np
import argparse
from skimage import measure
from evaluation_comparison.region_properties import RegionProperties
from evaluation_comparison.morphology import MorphologyOps
import nibabel as nib
from nifty_utils.file_utils import expand_to_5d, split_filename, reorder_list_presuf

MEASURES = ('centre of mass', 'volume', 'surface', 'surface volume ratio',
            'compactness', 'mean', 'weighted_mean', 'skewness',
            'kurtosis', 'min', 'max', 'std', 'quantile_1',
            'quantile_5', 'quantile_25', 'quantile_50',
            'quantile_75', 'quantile_95','quantile_99', 'asm', 'contrast',
            'correlation',
            'sumsquare',
            'sum_average', 'idifferentmomment', 'sumentropy', 'entropy',
            'differencevariance', 'sumvariance', 'differenceentropy',
            'imc1', 'imc2')
MEASURES_SIMPLE = ('centre of mass', 'volume', 'surface', 'surface volume ratio',
            'compactness', 'mean', 'weighted_mean', 'skewness',
            'kurtosis', 'min', 'max', 'std', 'quantile_1',
            'quantile_5', 'quantile_25', 'quantile_50',
            'quantile_75', 'quantile_95','quantile_99')

MEASURES_SHAPE = ('centre of mass', 'volume', 'surface', 'surface volume '
                                                         'ratio',
                  'compactness', 'solidity', 'balance', 'fractal_dim',
                  'circularity', 'contour_smoothness', 'eigen_values',
                  'ratio_eigen', 'fa')

OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'ROIStatistics'
OUTPUT_FIG_PREFIX = 'CumHist'


def extract_region_properties(imagename, maskname,
                              threshold=0.5, measures=MEASURES,
                              n_bins=100, mul=10, trans=50, type='file',
                              pixdim=[1,1,1]):
    """
    Extract region properties based on the intensity transformations and the
    measure list in measures
    :param imagename: image file to use to extract region properties
    :param maskname: image file to use as mask over the image of interest
    :param threshold: threshold to use over the mask to create the binary
    segmentation
    :param measures: list of measures to extract
    :param n_bins: number of bins to consider when performing haralick
    features extraction
    :param mul: multiplication factor to apply over the image
    :param trans: translation factor to apply over the image intensities
    :return: region properties, count and bins
    """
    if type == 'file':
        image_nii = nib.load(imagename)
        img = np.nan_to_num(expand_to_5d(image_nii.get_data()))
        mask_nii = nib.load(maskname)
        mask = mask_nii.get_data()
        pixdim = image_nii.get_header().get_zooms()

    else:
        img = np.nan_to_num(expand_to_5d(imagename))
        mask = maskname
    mask[mask > threshold] = 1

    mask[mask < threshold] = 0
    print(np.count_nonzero(mask), "non zero in mask")
    foreground_selector = np.where((mask > 0).reshape(-1))[0]
    img_flatten = img.reshape(-1)[
        foreground_selector]

    probs = mask.reshape(-1)[foreground_selector]

    print(np.min(img), np.max(img), "range of image")

    rp = RegionProperties(mask, img, measures, pixdim=pixdim, mul=mul,
                          trans=trans)

    foreground_selector = np.where((mask > 0).reshape(-1))[0]
    probs = mask.reshape(-1)[foreground_selector]
    img_flatten = img[..., 0, 0].reshape(-1)[foreground_selector]
    # plot the cumulative histogram
    img_flatten = np.nan_to_num(img_flatten)
    n, bins = np.histogram(img_flatten, n_bins)

    return rp, n, bins


def main(argv):

    parser = argparse.ArgumentParser(description='Create csv file with'
                                                 ' region properties')
    parser.add_argument('-i', dest='input_image', metavar='input pattern',
                        type=str,
                        help='RegExp pattern for the input files')
    parser.add_argument('-m', dest='mask_image', action='store',
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
    parser.add_argument('-name', dest='name', default='',
                        action='store',  type=str, help='name for filename')

    try:
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))
    except argparse.ArgumentTypeError:
        print('compute_ROI_statistics.py -i <input_image_pattern> -m '
              '<mask_image_pattern> -t <threshold> -mul <analysis_type> '
              '-trans <offset>   ')
        sys.exit(2)

    print(args.input_image, args.mask_image, args.trans, args.mul,
          args.threshold)
    images = glob.glob(args.input_image)
    masks = glob.glob(args.mask_image)
    print(images, masks)
    pth, name, ext = split_filename(images[0])
    if not args.name == '':
        name = args.name
    out_name = '{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        name)
    fig_name = '{}_{}.png'.format(
        OUTPUT_FIG_PREFIX,
        name)
    iteration = 0

    while os.path.exists(os.path.join(pth, out_name)):
        iteration += 1
        out_name = '{}_{}_{}.csv'.format(
            OUTPUT_FILE_PREFIX,
            name,
            str(iteration))

    out_stream = open(os.path.join(pth, out_name), 'a+')
    print("Writing {} to {}".format(out_name, pth))
    if args.threshold is None:
        args.threshold = 0.5
    if args.mul is None:
        args.mul = 10
    if args.trans is None:
        args.trans = 50
    print(args.threshold, args.mul, args.trans)
    img = nib.load(images[0]).get_data()
    img_2 = np.nan_to_num(expand_to_5d(img))
    if len(args.measures) == 1 and args.measures[0] == 'simple':
        argmeasures = MEASURES_SIMPLE
    if len(args.measures) == 1 and args.measures[0] == 'full':
        argmeasures = MEASURES
    if len(args.measures) == 1 and args.measures[0] == 'shape':
        argmeasures = MEASURES_SHAPE
    if len(args.measures) > 1:
        argmeasures = args.measures
    header_str = RegionProperties(img, img_2, argmeasures).header_str()
    fixed_fields = 'Mask,Image'
    if args.analysis != 'binary':
        fixed_fields = 'Mask,Image,Label'
    out_stream.write(fixed_fields + header_str + '\n')

    if args.analysis == 'cc' and args.measures[0] == 'shape':
        mask_names_init = glob.glob(args.mask_image)
        for mask_file in mask_names_init:
            mask = nib.load(mask_file).get_data()
            cc_map = measure.label(mask, connectivity=args.neighborhood,
                               background=0)
            values_label = np.unique(cc_map)
            values_label = [v for v in values_label if v > 0]
            for val in values_label:
                mask_label = np.where(cc_map == val, np.ones_like(mask),
                                  np.zeros_like(mask))

                roi_stats, n, bins = extract_region_properties(
                    mask, mask_label, threshold=args.threshold,
                 mul=args.mul, measures=argmeasures,
                    trans=args.trans, type='image')

                fixed_fields = '{},{},{}'.format(mask_file, mask_file, val)

                out_stream.write(fixed_fields + roi_stats.to_string(
                    OUTPUT_FORMAT) + '\n')

    elif args.type == 'paired':
        # inputs
        img_names_init = glob.glob(args.input_image)
        mask_names_init = glob.glob(args.mask_image)
        img_names = []
        mask_names = []
        # seg_names = util.list_files(param.seg_dir, param.ext)
        # ref_names = util.list_files(param.ref_dir, param.ext)
        ind_s, ind_r = reorder_list_presuf(img_names_init, mask_names_init)
        print(len(ind_s))
        for i in range(0, len(ind_s)):
            if ind_s[i] > -1:
                print(i, ind_s[i])
                print(img_names_init[i], mask_names_init[
                    ind_s[i]])
                img_names.append(img_names_init[i])
                mask_names.append(mask_names_init[ind_s[i]])
        pair_list = list(zip(img_names, mask_names))
        for pair in pair_list:
            image_file = pair[0]
            mask_file = pair[1]
            image = nib.load(image_file).get_data()
            mask = nib.load(mask_file).get_data()
            if args.analysis == 'binary':
                roi_stats, n, bins = extract_region_properties(
                    image, mask, threshold=args.threshold, mul=args.mul,
                    trans=args.trans, measures=argmeasures)
                pth, name, ext = split_filename(mask)
                fig_name = os.path.join(pth, '{}_{}_{}.png'.format(
                    OUTPUT_FIG_PREFIX,
                    name,
                    str(iteration)))
                import matplotlib.pyplot as plt
                if np.sum(n) > 0.1:
                    print(bins, np.sum(n))
                    fig, ax = plt.subplots(figsize=(4, 4))
                    cumsum = np.nan_to_num(np.cumsum(n)/(1.0*np.sum(n)))
                    ax.hist(bins[:-1], len(bins)-1, weights=cumsum, histtype='step')
                    ax.set_xlabel('Z-score')
                    ax.set_ylabel('Cumulative frequency')
                #    plt.show()
                    plt.savefig(fig_name)
                fixed_fields = '{},{}'.format(mask, image)
            if args.analysis == 'label':
                values_label = np.unique(mask)
                values_label = [v for v in values_label if v > 0]
                for val in values_label:
                    mask_label = np.where(mask == val, np.ones_like(mask),
                                          np.zeros_like(mask))
                    roi_stats, n, bins = extract_region_properties(
                        image, mask_label, threshold=args.threshold,
                        mul=args.mul, measures=argmeasures,
                        trans=args.trans, type='image')

                    fixed_fields = '{},{},{}'.format(mask_file, image_file, val)

                    out_stream.write(fixed_fields + roi_stats.to_string(
                        OUTPUT_FORMAT) + '\n')
            if args.analysis == 'cc':

                cc_map = measure.label(mask, connectivity=args.neighborhood,
                                       background=0)
                values_label = np.unique(cc_map)
                values_label = [v for v in values_label if v > 0]
                for val in values_label:
                    mask_label = np.where(cc_map == val, np.ones_like(mask),
                                          np.zeros_like(mask))

                    roi_stats, n, bins = extract_region_properties(
                        image, mask_label, threshold=args.threshold,
                        mul=args.mul, measures=argmeasures,
                        trans=args.trans, type='image')

                    fixed_fields = '{},{},{}'.format(mask_file, image_file, val)

                    out_stream.write(fixed_fields + roi_stats.to_string(
                        OUTPUT_FORMAT) + '\n')
    else:
        image = images[0]
        for mask in masks:
            roi_stats, n, bins = extract_region_properties(
                image, mask, threshold=args.threshold, mul=args.mul,
                trans=args.trans)
            import matplotlib.pyplot as plt
            if np.sum(n) > 1:
                print(bins)
                fig, ax = plt.subplots(figsize=(4, 4))
                cumsum = np.nan_to_num(np.cumsum(n) / (1.0 * np.sum(n)))
                ax.hist(bins[:-1], len(bins) - 1, weights=cumsum,
                        histtype='step')
                ax.set_xlabel('Z-score')
                ax.set_ylabel('Cumulative frequency')
            # plt.show()
                plt.savefig(fig_name)
            fixed_fields = '{},{}'.format(mask, image)

            out_stream.write(fixed_fields + roi_stats.to_string(
                OUTPUT_FORMAT) + '\n')

    out_stream.close()

if __name__ == "__main__":
    main(sys.argv[1:])




