from __future__ import absolute_import, print_function

import os.path
import sys
import getopt
import glob
import numpy as np
import argparse
from evaluation_comparison.region_properties import RegionProperties
import nibabel as nib
from nifty_utils.file_utils import expand_to_5d, split_filename

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
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'ROIStatistics'
OUTPUT_FIG_PREFIX = 'CumHist'


def extract_region_properties(imagename, maskname,
                              threshold=0.5, measures=MEASURES,
                              n_bins=100, mul=10, trans=50):
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
    image_nii = nib.load(imagename)
    mask_nii = nib.load(maskname)
    mask = mask_nii.get_data()
    mask[mask > threshold] = 1
    mask[mask < threshold] = 0
    print(np.count_nonzero(mask), "non zero in mask")
    img = np.nan_to_num(expand_to_5d(image_nii.get_data()))
    foreground_selector = np.where((mask > 0).reshape(-1))[0]
    probs = mask.reshape(-1)[foreground_selector]
    img_flatten = image_nii.get_data().reshape(-1)[
                                         foreground_selector]
    print(np.min(image_nii.get_data()), np.max(image_nii.get_data()), "range "
                                                                      "of "
                                                                      "image")
    pixdim = image_nii.get_header().get_zooms()
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
                        type=str, required=True,
                        help='RegExp pattern for the input files')
    parser.add_argument('-m', dest='mask_image', action='store',
                        default='', type=str,
                        help='RegExp pattern for the mask files')
    parser.add_argument('-t', dest='threshold', action='store', default=0.5,
                        type=float, help='threshold to apply to get a binary '
                                         'mask')
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

    print(args.input_image, args.mask_image, args.trans, args.mul,
          args.threshold)
    images = glob.glob(args.input_image)
    masks = glob.glob(args.mask_image)
    print(images, masks)
    pth, name, ext = split_filename(images[0])
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
    header_str = RegionProperties(img, img_2, MEASURES).header_str()
    out_stream.write('Mask, Image' + header_str + '\n')
    if len(images) == len(masks):
        for (image, mask) in zip(images, masks):
            roi_stats, n, bins = extract_region_properties(
                image, mask, threshold=args.threshold, mul=args.mul,
                trans=args.trans)
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
            # plt.show()
                plt.savefig(fig_name)
            fixed_fields = '{},{}'.format(mask, image)

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




