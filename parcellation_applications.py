import argparse
import os
import sys
import glob
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion, \
    binary_fill_holes, distance_transform_edt

DICT_VASCULAR = {'MCA_Left': 22 , 'MCA_Right': 21, 'PCA_Left': 12, 'PCA_Right':
11,
'ACA_Right': 31,'ACA_Left': 32}
DICT_STRUCT = {'FrontalLeft': 1, 'FrontalRight': 2, 'ParietalLeft':3,
               'ParietalRight':4, 'OccipitalLeft':5, 'OccipitalRight':6,
               'TemporalLeft':7, 'TemporalRight':8, 'BasalGanglia':9,
               'Infratentorial':10}

from parcellation_utils.parcellation_aggregate import combine_seg,\
    prepare_use_gif_hierarchy, \
    create_hemisphere, create_bg, create_aggregation, create_aggregated_volume
from parcellation_utils.parcellation_parsing import get_dict_match, \
    get_dict_parc, get_hierarchy


def main(argv):
    association_file = os.path.join(os.path.split(os.path.abspath(
        __file__))[0], 'parcellation_utils', 'GIFHierarchy.csv')
    association_file_dbgif = os.path.join(os.path.split(os.path.abspath(
        __file__))[0], 'parcellation_utils', 'KeysHierarchy_ordered.csv')
    lobesfile = os.path.join(os.path.split(os.path.abspath(
        __file__))[0], 'parcellation_utils', 'TerritoriesLobesMapping.csv')
    demographic_file = None
    pattern = "*.xml"
    exclusion = "zzzzzzzz"
    strip_name_right = '_NeuroMorph.xml'
    strip_name_left = ''

    parser = argparse.ArgumentParser(description='Create parcellation based '
                                                 'segmentation aggregations')
    parser.add_argument('-f', dest='file_pattern', metavar='filename_input',
                        type=str, required=True,
                        help='file where the input parcellation is located')
    parser.add_argument('-p', dest='output_path', action='store',
                        default=os.getcwd(),
                        help='output_path')
    subparsers = parser.add_subparsers(dest='subcommand')

    #  subparser for checks on orientation and acquisition
    parser_check = subparsers.add_parser('checks')
    parser_check.add_argument('-iso', dest='iso_flag', action='store_true')
    parser_check.add_argument('-ori', dest='ori_flag', action='store_true')

    # subparser for aggregations based on label hierarchy
    parser_seg = subparsers.add_parser('seg_aggregate')
    parser_seg.add_argument('-bg', dest='bg_flag', action='store_true')
    parser_seg.add_argument('-hemi', dest='hemi_flag', action='store_true')

    parser_seg.add_argument('-a', type=str, dest='aggregation_list',
                            action='store',
                            help="Indicate which structure should be segmented "
                                 "or list of structures separated by , "
                                 "ex: (\"Frontal Lobe\" or "
                                 "\"Frontal Lobe, Parietal Lobe\"")
    parser_seg.add_argument('-combi', type=str, choices=['separated',
                                                         'split_4d',
                                                         'label_3d',
                                                         'combined_binary'],
                            default='combined_binary', dest='combination',
                            help="Indicates how multiple structures should "
                                 "be eventually combined")
    parser_seg.add_argument('-look_up', dest='look_up', action='store',
                            type=str)


    parser_lobes = subparsers.add_parser('lobes')
    parser_lobes.add_argument('-s', dest='split', action='store_true')
    parser_lobes.add_argument('-laplace', dest='laplace_file',
                              action='store', type=str)
    parser_lobes.add_argument('-a', dest='assign', choices=['euc', 'lap'],
                              default='euc')
    parser_lobes.add_argument('-m', dest='mask', type=str)

    # subparser for parsing of xml file of the parcellation file or if no xml
    #  file, creating the volumetric database based on all possible labels
    # and their combination
    parser_parsing = subparsers.add_parser('parsing')

    parser_parsing.add_argument('-a', dest='association_file',
                                action='store',
                                help='File for the association to GIF output',
                                type=str, default=association_file)

    parser_parsing.add_argument('-o', dest='output_file', action='store',
                                help='Where to write output for the database',
                                type=str)
    parser_parsing.add_argument('-l', dest='left_strip', action='store',
                                help='What to strip from input file name '
                                     'on the left ',
                                type=str, default=strip_name_left)
    parser_parsing.add_argument('-r', dest='right_strip', action='store',
                                type=str,
                                default=strip_name_right,
                                help='What to strip from input file name '
                                     'on the right')
    parser_parsing.add_argument('-d', dest='demographic_file',
                                action='store',
                                type=str,
                                help='demographic file to further match '
                                     'individuals', default=None)
    parser_parsing.add_argument('-e', dest='exclude', action='store',
                                default=exclusion, type=str)

    # to build database from parcellation files
    parser_dbnii = subparsers.add_parser('database_fromparc')
    parser_dbnii.add_argument('-a', dest='association_file',
                              action='store',
                              help='File for the association to GIF output',
                              type=str, default=association_file_dbgif)
    parser_dbnii.add_argument('-o', dest='output_file', action='store',
                              help='Where to write output for the database',
                              type=str)
    parser_dbnii.add_argument('-l', dest='left_strip', action='store',
                              help='What to strip from input file name'
                                   ' on the left ', type=str,
                              default=strip_name_left)
    parser_dbnii.add_argument('-r', dest='right_strip', action='store',
                              type=str, default=strip_name_right, help='What '
                              'to strip from input file name on the right')

    try:
        args = parser.parse_args()
        # print(args.accumulate(args.integers))

    except argparse.ArgumentTypeError:
        print('BrainHearts.py -f <filename_database> -g <grouping> -d '
              '<dependent variable> -i <independent variables>')
        print('The list of independent variables must always start with the '
              'Age')
        sys.exit(2)

    list_files = glob.glob(args.file_pattern)
    print(len(list_files))

    if args.subcommand == 'lobes':

        df_parc = pd.DataFrame.from_csv(lobesfile)
        val_terr = np.unique(df_parc['FullTerr'])
        val_lobe = np.unique(df_parc['FullStruct'])
        if not args.split:
            for f in list_files:
                parc_nii = nib.load(f)
                parc_data = parc_nii.get_data()
                lobar_separation = np.zeros_like(parc_data)
                terr_separation = np.zeros_like(parc_data)
                for val in val_terr:
                    if val>0:
                        df_select = df_parc[df_parc['FullTerr']==val]
                        val_gif = np.unique(df_select['GIF'])
                        seg_temp = np.where(parc_data in val_gif, np.ones_like(
                            parc_data) * val, np.zeros_like(parc_data))
                        terr_separation += seg_temp
                for val in val_lobe:
                    if val>0:
                        df_select = df_parc[df_parc['FullStruct']==val]
                        val_gif = np.unique(df_select['GIF'])
                        seg_temp = np.where(parc_data in val_gif, np.ones_like(
                            parc_data) * val, np.zeros_like(parc_data))
                        if val==9:
                            seg_temp = seg_temp.astype(bool)
                            seg_temp = binary_dilation(seg_temp, iterations=4)
                            seg_temp = binary_fill_holes(seg_temp)
                            seg_temp = binary_erosion(seg_temp, iterations=4)
                            seg_temp = seg_temp.astype(float)*9

                        lobar_separation += seg_temp
                name_lobes = 'Lobes_' + os.path.split(f)[1]
                name_terr = 'Territories_' + os.path.split(f)[1]
                name_lobes = os.path.join(args.output_path, name_lobes)
                name_terr = os.path.join(args.output_path, name_terr)
                lobes_nii = nib.Nifti1Image(lobar_separation, parc_nii.affine)
                terr_nii = nib.Nifti1Image(terr_separation, parc_nii.affine)
                nib.save(terr_nii, name_terr)
                nib.save(lobes_nii, name_lobes)

        # For now the following is only perform for one given subject and
        # does not support pairing of multiple files:
        if args.split:
            f = list_files[0]
            parc_nii = nib.load(list_files[0])
            parc_data = parc_nii.get_data()
            zooms = parc_nii.header.get_zooms()
            if args.mask is None:
                mask = (parc_data > 12).astype(int)
                ventr_data = (parc_data < 54).astype(int) * (parc_data >
                                                         49).astype(int)
                mask -= ventr_data
            else:
                mask = nib.load(args.mask).get_data()
            lobar_separation = np.zeros_like(parc_data)
            terr_separation = np.zeros_like(parc_data)
            list_dist_lobes = []
            list_dist_terr = []
            for val in val_terr:
                if val > 0:
                    df_select = df_parc[df_parc['FullTerr'] == val]
                    val_gif = np.unique(df_select['GIF'])
                    seg_temp = np.zeros_like(parc_data)
                    for gv in val_gif:
                        seg_temp = np.where(parc_data==gv, np.ones_like(
                        parc_data) , seg_temp)
                    terr_separation += seg_temp * val
                    print(np.sum(seg_temp))
                    distance_terr = distance_transform_edt(seg_temp * -1 +1,
                                                           sampling=zooms)
                    list_dist_terr.append(np.expand_dims(distance_terr *
                                                         mask, -1))
            for val in val_lobe:
                if val > 0:
                    df_select = df_parc[df_parc['FullStruct'] == val]
                    val_gif = np.unique(df_select['GIF'])

                    seg_temp = np.zeros_like(parc_data)
                    for gv in val_gif:
                        seg_temp = np.where(parc_data == gv, np.ones_like(
                            parc_data), seg_temp)

                    if val == 9:
                        seg_temp = seg_temp.astype(bool)
                        seg_temp = binary_dilation(seg_temp, iterations=4)
                        seg_temp = binary_fill_holes(seg_temp)
                        seg_temp = binary_erosion(seg_temp, iterations=4)
                        seg_temp = seg_temp.astype(float)

                    lobar_separation += seg_temp * val
                    if val < 9:
                        distance_lobe = distance_transform_edt(seg_temp * -1 +1,
                                                           sampling=zooms)
                        list_dist_lobes.append(np.expand_dims(distance_lobe *
                                               mask, -1))
            stacked_dist_terr = np.concatenate(list_dist_terr, -1)
            stacked_dist_lobes = np.concatenate(list_dist_lobes, -1)
            final_assign_terr = (np.argmin(stacked_dist_terr, -1) + 1) * mask
            final_assign_lobes = (np.argmin(stacked_dist_lobes, -1) +1) *mask
            final_assign_lobes = np.where(lobar_separation>8,
                                          lobar_separation, final_assign_lobes)
            name_lobes = 'Lobes_' + os.path.split(f)[1]
            name_terr = 'Territories_' + os.path.split(f)[1]
            name_asslobes = 'AssignLobes_' + os.path.split(f)[1]
            name_assterr = 'AssignTerritories_' + os.path.split(f)[1]
            name_lobes = os.path.join(args.output_path, name_lobes)
            name_terr = os.path.join(args.output_path, name_terr)
            name_asslobes = os.path.join(args.output_path, name_asslobes)
            name_assterr = os.path.join(args.output_path, name_assterr)
            lobes_nii = nib.Nifti1Image(lobar_separation, parc_nii.affine)
            terr_nii = nib.Nifti1Image(terr_separation, parc_nii.affine)
            nib.save(terr_nii, name_terr)
            nib.save(lobes_nii, name_lobes)
            asslobes_nii = nib.Nifti1Image(final_assign_lobes, parc_nii.affine)
            assterr_nii = nib.Nifti1Image(final_assign_terr, parc_nii.affine)
            nib.save(assterr_nii, name_assterr)
            nib.save(asslobes_nii, name_asslobes)






    if args.subcommand == 'parsing':

        if args.demographic_file is not None:
            demographic_df = pd.DataFrame.from_csv(path=demographic_file)
            demographic_dict = demographic_df.to_dict()
        else:
            demographic_dict = {}
        path_results = args.output_file
        dict_hierarchy = get_hierarchy(args.association_file)
        list_parcellation = glob.glob(args.file_pattern)
        test = get_dict_parc(list_parcellation[0])
        dict_new = get_dict_match(test, dict_hierarchy)
        list_keys_columns = dict_new.keys()
        sorted_keys = sorted(list_keys_columns)
        columns = ['ID'] + list(demographic_dict.keys()) + ['TIV'] + sorted_keys
        dict_total = {c: [] for c in columns}
        print("Number of files to process is %d" % len(list_parcellation))
        for parc in list_parcellation:
            name = os.path.split(parc)[1].rstrip(args.right_strip)
            name = name.lstrip(args.left_strip)
            print(name)
            if 'DOB' in demographic_dict.keys():
                if args.exclude not in parc and name in \
                        demographic_dict['DOB'].keys():
                    dict_temp = get_dict_parc(parc)
                    dict_fin = get_dict_match(dict_temp, dict_hierarchy)
                    dict_fin['File'] = parc
                    tiv = 0
                    for col in list_keys_columns:
                        if '6_' in col and col not in ('6_0', '6_1', '6_2',
                                                       '6_3','6_4'):
                            tiv += float(dict_fin[col])
                        dict_total[col].append(dict_fin[col])
                    dict_total['TIV'].append(tiv)
                    if name in demographic_dict['DOB'].keys():
                        dict_total['ID'].append(name)
                        for demkeys in demographic_dict.keys():
                            if demkeys == 'sex':
                                dict_total[demkeys].append(
                                    demographic_dict[demkeys][name] - 1)
                            else:
                                dict_total[demkeys].append(
                                    demographic_dict[demkeys][name])
            else:
                dict_temp = get_dict_parc(parc)
                dict_fin = get_dict_match(dict_temp, dict_hierarchy)
                dict_fin['File'] = parc
                dict_total['ID'].append(name)
                tiv = 0
                for col in list_keys_columns:
                    if '6_' in col and col not in ('6_0', '6_1', '6_2',
                                                   '6_3','6_4'):
                        tiv += float(dict_fin[col])
                    dict_total[col].append(dict_fin[col])
                dict_total['TIV'].append(tiv)

        df_tot = pd.DataFrame(dict_total)
        df_tot.to_csv(path_results, header=True, columns=columns)

    if args.subcommand == 'database_fromparc':
        gif_h, dict_levels = prepare_use_gif_hierarchy()
        list_dict_parc = []
        for filename in list_files:
            print("Processing %s" % filename)
            name = os.path.split(filename)[1]
            name = name.rstrip(args.right_strip)
            name = name.lstrip(args.left_strip)
            parc = nib.load(filename)
            parc_data = parc.get_data()
            dict_temp = {'Name': name}
            pixdim = parc.header.get_zooms()
            volume_voxel = pixdim[0] * pixdim[1] * pixdim[2]
            for agg in dict_levels.keys():
                vol_temp = create_aggregated_volume(parc_data, agg, gif_h,
                                                    dict_levels)
                dict_temp[a] = vol_temp * volume_voxel
            list_dict_parc.append(dict_temp)
        pd_parc = pd.DataFrame.from_dict(list_dict_parc)
        pd_parc.to_csv(args.output_file)

    if args.subcommand == 'seg_aggregate':
        aggregation = None
        gif_h = None
        dict_levels = None
        # first do the checks on aggregation wanted
        if args.aggregation_list is not None:
            aggregation = args.aggregation_list.split(',')
            aggregation = [agg.strip(' ') for agg in aggregation]
            gif_h, dict_levels = prepare_use_gif_hierarchy()

        for filename in list_files:
            name = os.path.split(filename)[1]
            parc = nib.load(filename)
            parc_affine = parc.affine
            parc_data = parc.get_data()
            if args.bg_flag:
                bg_nii = create_bg(filename)
                nib.save(bg_nii, os.path.join(args.output_path, 'DGM_%s' %
                                              name))
            if args.hemi_flag:
                right_nii, left_nii = create_hemisphere(filename)
                nib.save(right_nii, os.path.join(args.output_path,
                                                 'RightHemi_%s' % name))
                nib.save(left_nii, os.path.join(args.output_path,
                                                'LeftHemi_%s') % name)
            if aggregation is not None:
                seg_aggregate = []
                for a in aggregation:
                    temp_seg = create_aggregation(parc_data, a, gif_h,
                                                  dict_levels)
                    seg_aggregate.append(temp_seg)
                final_seg = combine_seg(seg_aggregate, args.combination)
                if args.combination == 'separated':
                    for (final, agg) in zip(final_seg, aggregation):
                        nii_f = nib.Nifti1Image(final, parc_affine)
                        nib.save(nii_f, os.path.join(args.output_path,
                                                     '%s_%s') % (agg, name))
                else:
                    nii_f = nib.Nifti1Image(final_seg[0], parc_affine)
                    name_save = ''.join(aggregation)
                    name_save = name_save.replace(' ', '')
                    nib.save(nii_f, os.path.join(args.output_path, '%s_%s') %
                             (name_save, name))


if __name__ == "__main__":
    main(sys.argv[1:])