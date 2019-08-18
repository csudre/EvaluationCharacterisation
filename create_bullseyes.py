import argparse
import sys
import os
import getopt
import matplotlib.pyplot as plt
from bullseyes.bullseye_plotting import read_ls_create_agglo, \
    create_bullseye_plot, FULL_LABELS, FULL_LABELS_IT, \
    prepare_data_fromagglo
from bullseyes.creation_ls_file import creation_ls, write_ls
from nifty_utils.file_utils import split_filename



def main(argv):

    parser = argparse.ArgumentParser(description='Create database of local '
                                                 'summaries ')
    parser.add_argument('-f', dest='filewrite', metavar='file where to write',
                        type=str, required=True,
                        help='File where to write result')
    parser.add_argument('-l', dest='layer_file', action='store',
                        help='Layer image to use', type=str)
    parser.add_argument('-o', dest='lobar_file', action='store',
                        help='Lobar image to use', type=str)
    parser.add_argument('-nl', dest='numb_layers', action='store', default=4,
                        type=int)
    parser.add_argument('-p', dest='path_result', action='store',
                        help='Path where to save the results', type=str,
                        default=os.getcwd())
    parser.add_argument('-ci', dest='corr_it', action='store_true',
                        help='indicates if we should correct for the '
                             'infratentorial region')
    parser.add_argument('-n', dest='name', action='store', default='',
                        help='name to use for the saving of the bullseye plot')

    try:
        args = parser.parse_args(argv)

    except getopt.GetoptError:
        print('creation_ls_file.py -l <layer_file> -f <filename_write> -o '
              '<lobar_file> -s '
              '<segmentation_file> ')
        sys.exit(2)

    if args.lobe_file is not None and args.layer_file is not None and \
            args.lesion_file is not None:
        vol_prob, vol_bin, vol_reg, connect = creation_ls(args.lobe_file,
                                                          args.layer_file,
                                                          args.lesion_file)
        if args.filewrite is None:
            [path, basename, _] = split_filename(args.lesion_file)
            filewrite = os.path.join(path, 'LocalSummary_'+basename+'.txt')
        write_ls(vol_prob, vol_bin, vol_reg, connect, args.filewrite)
    if args.filewrite is not None:
        les, reg, freq, dist = read_ls_create_agglo(args.filewrite)
        freq_full = prepare_data_fromagglo(freq, type_prepa="full")
        create_bullseye_plot(freq_full, 'YlOrRd', num_layers=args.numb_layers,
                             num_lobes=9, vmin=0, vmax=0.25,
                             labels=FULL_LABELS)
        plt.savefig(os.path.join(args.path_result, 'BE_'+args.name+'.png'))
    print("printednow")


if __name__ == "__main__":
    main(sys.argv[1:])

