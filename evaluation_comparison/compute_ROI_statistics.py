from __future__ import absolute_import, print_function

import os.path
import sys
import getopt
import glob
import numpy as np
import region_properties
import nibabel as nib

MEASURES = ('centre of mass', 'volume', 'surface', 'surface volume ratio',
            'compactness', 'mean', 'weighted_mean', 'skewness',
            'kurtosis', 'min', 'max', 'std', 'quantile_1',
            'quantile_5','quantile_25', 'quantile_50',
            'quantile_75', 'quantile_95','quantile_99', 'asm', 'contrast',
            'correlation',
            'sumsquare',
            'sum_average', 'idifferentmomment', 'sumentropy', 'entropy',
            'differencevariance', 'sumvariance', 'differenceentropy',
            'imc1', 'imc2')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'ROIStatistics'
OUTPUT_FIG_PREFIX = 'CumHist'


def split_filename(file_name):
    '''
    Operation on filename to separate path, basename and extension of a filename
    :param file_name: Filename to treat
    :return pth, fname, ext:
    '''
    pth = os.path.dirname(file_name)
    fname = os.path.basename(file_name)

    ext = None
    for special_ext in '.nii', '.nii.gz':
        ext_len = len(special_ext)
        if fname[-ext_len:].lower() == special_ext:
            ext = fname[-ext_len:]
            fname = fname[:-ext_len] if len(fname) > ext_len else ''
            break
    if ext is None:
        fname, ext = os.path.splitext(fname)
    return pth, fname, ext


def expand_to_5d(img_data):
    '''
    Expands an array up to 5d if it is not the case yet
    :param img_data:
    :return:
    '''
    while img_data.ndim < 5:
        img_data = np.expand_dims(img_data, axis=-1)
    return img_data


def extract_region_properties(imagename, maskname,
                              threshold=0.5, measures=MEASURES,
                              n_bins=100, mul=10, trans=50):
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
    rp = region_properties.RegionProperties(mask, img, measures,
                                            pixdim=pixdim, mul=mul, trans=trans)

    foreground_selector = np.where((mask > 0).reshape(-1))[0]
    probs = mask.reshape(-1)[foreground_selector]
    img_flatten = img[..., 0, 0].reshape(-1)[foreground_selector]
    # plot the cumulative histogram
    img_flatten = np.nan_to_num(img_flatten)
    n, bins = np.histogram(img_flatten, n_bins)

    return rp, n, bins


def main(argv):
    inputimage = ''
    maskimage = ''
    threshold = None
    mul=None
    trans=None
    try:
        opts, args = getopt.getopt(argv, "hi:m:t:u:r:", ["ifile=",
                                                            "mfile=",
                                                            "trans=","mul="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -m <maskfile> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <regexp> -m <regexp> ')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputimage = arg
        elif opt in ("-m", "--mfile"):
            maskimage = arg
        elif opt in ("-t", "--threshold"):
            threshold = float(arg)
        elif opt in ("-u","--mul"):
            mul=float(arg)
        elif opt in ("-r","--trans"):
            trans=float(arg)
    print(inputimage, maskimage, trans,mul,threshold)
    images = glob.glob(inputimage)
    masks = glob.glob(maskimage)
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
    if threshold is None:
        threshold=0.5
    if mul is None:
        mul=10
    if trans is None:
        trans=50
    print(threshold,mul,trans)
    img = nib.load(images[0]).get_data()
    img_2 = np.nan_to_num(expand_to_5d(img))
    header_str = region_properties.RegionProperties(img, img_2,
                                                    MEASURES).header_str()
    out_stream.write('Mask, Image' + header_str + '\n')
    if len(images)==len(masks):
        for (image, mask) in zip(images, masks):
            roi_stats, n, bins = extract_region_properties(image, mask,
                                                   threshold=threshold,
                                                       mul=mul, trans=trans)
            pth, name, ext = split_filename(mask)
            fig_name = os.path.join(pth,'{}_{}_{}.png'.format(
                OUTPUT_FIG_PREFIX,
                name,
                str(iteration)))
            import matplotlib.pyplot as plt
            if np.sum(n) > 0.1:
                print(bins, np.sum(n))
                fig, ax = plt.subplots(figsize=(4, 4))
                cumsum = np.nan_to_num(np.cumsum(n)/(1.0*np.sum(n)))
                ax.hist(bins[:-1], len(bins)-1, weights=cumsum,
                histtype='step')
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
            roi_stats, n, bins = extract_region_properties(image, mask,
                                                           threshold=threshold,
                                                           mul=mul, trans=trans)
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

