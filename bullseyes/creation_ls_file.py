import nibabel as nib
import numpy as np
import os
from scipy import ndimage
from bullseye_plotting import (create_bullseye_plot, prepare_data_bullseye,
                               read_ls_create_agglo, prepare_data_fromagglo,

                               LABELS_LR, FULL_LABELS)
import matplotlib.pyplot as plt
import sys
import getopt


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


def creation_ls(lobe_file, layer_file, lesion_file):
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
    layer = np.where(layer==max_layer, (max_layer-1)*np.ones_like(layer), layer)
    [connected, num_connect] = ndimage.measurements.label(lesion)
    for o in range(0, max_lobe):
        for l in range(1, max_layer):
            region_lobe = np.where(lobe == o+1, np.ones_like(lobe),
                                   np.zeros_like(lobe))
            region_layer = np.where(layer==l, np.ones_like(layer),
                                    np.zeros_like(layer))
            region = np.multiply(region_lobe, region_layer)
            lesion_region = np.multiply(region, lesion)
            vol_prob.append(np.sum(lesion_region)*vol_vox)
            vol_bin.append(np.where(lesion_region>0)[0].shape[0]*vol_vox)
            values = np.unique(connected*lesion_region)
            connect.append(len(values)-1)
            vol_reg.append(np.sum(region)*vol_vox)
    vol_prob.append(np.sum(lesion)*vol_vox)
    vol_bin.append(np.where(lesion > 0)[0].shape[0]*vol_vox)
    vol_reg.append(np.where(lobe*layer > 0)[0].shape[0]*vol_vox)
    connect.append(num_connect)
    return vol_prob, vol_bin, vol_reg, connect


def write_ls(vol_prob, vol_bin,vol_reg, connect, filewrite):
    with open(filewrite, 'w') as out_stream:
        for (vp, vb, vr, c) in zip(vol_prob, vol_bin, vol_reg, connect):
            out_stream.write(str(vp)+" "+ str(vb)+' '+str(vr)+' '+
                             str(c)+'\n')


def bullseyes_from_nii(lobe_file, layer_file, lesion_file, filewrite):
    vol_prob, vol_bin, vol_reg, connect = creation_ls(lobe_file, layer_file,
                                                      lesion_file)
    write_ls(vol_prob, vol_bin, vol_reg, connect, filewrite)
    VPerc, VDist = prepare_data_bullseye(filewrite)
    be_perc = create_bullseye_plot(VPerc, 'YlOrRd', 0, 0.25)
    plt.show()
    be_dist = create_bullseye_plot(VDist, 'seismic', 0, 0.1)
    plt.show()


def main(argv):
    lobe_file = None
    layer_file = None
    lesion_file = None
    filewrite = None
    pathsave = os.getcwd()
    num_layers = 4
    corr_it = False
    name = ''

    try:
        opts, args = getopt.getopt(argv, "hl:o:f:s:n:p:nl:ci:", ["lobe=",
                                                               "layer=",
                                                               "lesion=",
                                                       "file=", "name=",
                                                           "path="])
    except getopt.GetoptError:
        print('creation_ls_file.py -l <layer_file> -f <filename_write> -o '
              '<lobar_file> -s '
              '<segmentation_file> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('creation_ls_file.py -l <layer_file> -f <filename_write> -o '
              '<lobar_file> -s '
              '<segmentation_file> ')
            sys.exit()
        elif opt in ("-f", "--file"):
            filewrite = arg
        elif opt in ("-l", "--layer"):
            layer_file = arg
        elif opt in ("-o", "--lobe"):
            lobe_file = arg
        elif opt in ("-s", "--lesion"):
            lesion_file = arg
        elif opt in ("-nl", "--num_layers"):
            num_layers = int(arg)
        elif opt in ("-ci", "--correct_it"):
            corr_it = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-p", "--path"):
            pathsave = arg
    if lobe_file is not None and layer_file is not None and lesion_file is \
            not None:
        vol_prob, vol_bin, vol_reg, connect = creation_ls(lobe_file, layer_file,
                                                      lesion_file)
        if filewrite is None:
            [path, basename, ext] = split_filename(lesion_file)
            filewrite = os.path.join(path, 'LocalSummary_'+basename+'.txt')
        write_ls(vol_prob, vol_bin, vol_reg, connect, filewrite)
    if filewrite is not None:
        les, reg, freq, dist = read_ls_create_agglo(filewrite)
        freq_full = prepare_data_fromagglo(freq, type_prepa="full")
        # freq_lr = prepare_data_fromagglo(freq, type="lr")
        # be_lr = create_bullseye_plot(freq_lr, 'YlOrRd', num_layers=4,
        #                          num_lobes=5,
        #                          vmin=0,
        #                          vmax=1, labels=LABELS_LR, thr=0.1)
        # plt.show()
        be_full = create_bullseye_plot(freq_full, 'YlOrRd', num_layers=num_layers,
                                 num_lobes=9, vmin=0,
                         vmax=0.25, labels=FULL_LABELS)
        plt.savefig(os.path.join(pathsave,'BE_'+name+'.png'))
    print("printednow")


if __name__ == "__main__":
   main(sys.argv[1:])



