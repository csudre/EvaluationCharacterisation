
import argparse
import os
import glob
import getopt
import sys
import pandas as pd
import numpy as np
from nifty_utils.file_utils import reorder_list, split_filename
from bullseyes.creation_ls_file import write_ls, read_ls_create_agglo, \
    creation_ls
from bullseyes.creation_database_ls import create_header_foragglo, \
    create_header_foragglo_corr
from bullseyes.bullseye_plotting import read_ls_create_agglo, agglo_ls_without_speclobe


def main(argv):

    parser = argparse.ArgumentParser(description='Create database of local '
                                                 'summaries ')
    parser.add_argument('-f', dest='file_pattern', metavar='file pattern',
                        type=str, required=False, default='LocalSummary*',
                        help='Regexp pattern for the files with local summary')
    parser.add_argument('-s', dest='lesion_pattern', action='store',
                        help='Regexp pattern for the lesion files', type=str)
    parser.add_argument('-l', dest='layer_pattern', action='store',
                        help='Regexp pattern for the layer files', type=str)
    parser.add_argument('-o', dest='lobar_pattern', action='store',
                        help='Regexp pattern for the lobar files', type=str)
    parser.add_argument('-nl', dest='numb_layers', action='store', default=4,
                        type=int)
    parser.add_argument('-r', dest='result_file', action='store')
    parser.add_argument('-p', dest='path_result', action='store',
                        help='Path where to save the results', type=str,
                        default=os.getcwd())
    parser.add_argument('-ci', dest='corr_it', action='store_true',
                        help='indicates if we should correct for the '
                             'infratentorial region')
    try:
        args = parser.parse_args()
        # print(args.accumulate(args.integers))

    except getopt.GetoptError:
        print('creation_database_ls.py -l <layer_file_pattern> -f '
              '<filename_write_pattern> -o <lobar_file_patterm>'
              ' -s <segmentation_file> -r <result_file> '
              '-numb_layers num_layers -ci '
              'correction_infratentorial')
        sys.exit(2)

    if args.lesion_pattern is not None and args.layer_pattern is not None and \
            args.lobe_pattern is not None:
        lesion_list = glob.glob(args.lesion_pattern)
        lobe_list = glob.glob(args.lobe_pattern)
        layer_list = glob.glob(args.layer_pattern)
        ind_s, ind_o = reorder_list(lesion_list, lobe_list)
        ind_s2, ind_l = reorder_list(lesion_list, layer_list)
        lesion_fin = []
        layer_fin = []
        lobe_fin = []
        for i in range(0, len(ind_s)):
            if ind_s[i] > -1 and ind_o[i] > -1 and \
                    ind_s2[i] > -1 and ind_l[i] > -1:
                print(i, ind_s[i], ind_o[i])
                print(lesion_list[i], lobe_list[
                    ind_s[i]], layer_list[ind_s2[i]])
                lesion_fin.append(lesion_list[i])
                layer_fin.append(layer_list[ind_s2[i]])
                lobe_fin.append(lobe_list[ind_s[i]])
        triplet_list = list(zip(lesion_fin, lobe_fin, layer_fin))
        for (les, lobe, layer) in triplet_list:
            vol_prob, vol_bin, vol_reg, connect = creation_ls(lobe, layer, les)
            [_, basename, _] = split_filename(les)
            filewrite = os.path.join(args.path_result,
                                     args.file_pattern+basename+".txt")
            write_ls(vol_prob, vol_bin, vol_reg, connect, filewrite)

    list_files = glob.glob(args.file_pattern)
    print("Number of files for which to build database %d" %len(list_files))
    result_array = None
    for f in list_files:
        les_fin, reg_fin, freq_fin, dist_fin = read_ls_create_agglo(
            f, num_layers=args.numb_layers, corr_it=args.corr_it)
        les_fin2, reg_fin2, freq_fin2, dist_fin2 = agglo_ls_without_speclobe(
            les_fin, reg_fin, 4, lobe_remove=[])
        final_array = np.concatenate((les_fin, reg_fin, freq_fin,
                                      dist_fin, les_fin2, reg_fin2, freq_fin2,
                                      dist_fin2), 0).T
        [_, basename, _] = split_filename(f)
        final_array = [basename] + list(final_array)
        if result_array is None:
            result_array = [final_array]
        else:
            result_array = result_array + [final_array]
    header_columns = create_header_foragglo(args.numb_layers)
    header_bis = create_header_foragglo_corr(lobes=['F', 'P', 'O', 'T', 'BG',
                                                    'IT'])
    header_columns_tot = header_columns + header_bis
    data_pd = pd.DataFrame(result_array, columns=header_columns_tot)
    data_pd.to_csv(args.result_file)


if __name__ == "__main__":
    main(sys.argv[1:])