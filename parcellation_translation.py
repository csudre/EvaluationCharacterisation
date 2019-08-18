import os
import argparse
import sys
import nibabel as nib
from parcellation_utils.gif_to_fs import relabel_over_mask, map_gif_to_fs


def main(argv):

    parser = argparse.ArgumentParser(description='Transform GIF parcellation '
                                                 'into FS parcellation.')

    subparsers = parser.add_subparsers(dest='subcommand')

    #  subparser for gif_to_fs
    parser_giftofs = subparsers.add_parser('gif_to_fs')
    parser_giftofs.add_argument('-p', dest='parcellation',
                                metavar='filename_input',
                                type=str, required=True,
                                help='file where the input parcellation '
                                     'is located')
    parser_giftofs.add_argument('-id', dest='id_subject', action='store',
                                type=str, help='subject_name', default='pid')
    parser_giftofs.add_argument('-o', dest='output_name', action='store',
                                default=os.path.join(os.getcwd(),
                                                     'FSMappedParc.nii.gz'),
                                help='output_name')
    parser_giftofs.add_argument('-with_split', action='store_true',
                                dest='split_complete', )

    #  subparser for mask_mapping
    parser_maskmap = subparsers.add_parser('mask_mapping')
    parser_maskmap.add_argument('-p', dest='parcellation',
                                metavar='filename_input',
                                type=str, required=True,
                                help='file where the input parcellation '
                                     'is located')
    parser_maskmap.add_argument('-id', dest='id_subject', action='store',
                                type=str,
                                help='subject_name', default='pid')
    parser_maskmap.add_argument('-o', dest='output_name', action='store',
                                default=os.path.join(os.getcwd(),
                                                     'FSMappedParc.nii.gz'),
                                help='output_name')

    parser_maskmap.add_argument('-m', dest='mask', metavar='filename_mask',
                                type=str, required=True,
                                help='file where the mask is located')
    parser_maskmap.add_argument('-l', dest='label', metavar='filename_label',
                                type=str, required=True,
                                help='file where the new label to apply over '
                                     'mask is located')

    try:
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))

    except argparse.ArgumentTypeError:
        print('BrainHearts.py -f <filename_database> -g <grouping> -d '
              '<dependent variable> -i <independent variables>')
        print('The list of independent variables must always start with the '
              'Age')
        sys.exit(2)

    if args.subcommand == 'gif_to_fs':
        new_parc = map_gif_to_fs(args.parcellation)
        if args.output_name is None:
            name_save = os.path.join(args.output_path,
                                     'FSMappedParc_'+args.id_subject+'.nii.gz')
        else:
            name_save = args.output_name
        nib.save(new_parc, name_save)

    if args.subcommand == 'mask_mapping':
        new_parc = relabel_over_mask(args.parcellation, args.mask, args.label)
        if args.output_name is None:
            name_save = os.path.join(args.output_path,
                                     'NewMapping_'+args.id_subject+'.nii.gz')
        else:
            name_save = args.output_name
        nib.save(new_parc, name_save)

if __name__ == "__main__":
   main(sys.argv[1:])

