import argparse
import os
import sys
import glob
import numpy as np
import nibabel as nib
from nifty_utils.orientation import nr_affine_to_flirt, flirt_affine_to_nr, \
    do_reorientation, check_coronal, check_axial, check_anisotropy, \
    check_sagittal, four_to_five, five_to_four, save_sform
CHOICES = ['LAS', 'LAI', 'LPS', 'LPI', 'LSA', 'LSP', 'LIA', 'LIP',
           'RAS', 'RAI', 'RPS', 'RPI', 'RSA', 'RSP', 'RIA', 'RIP',
           'SRP', 'SRA', 'SLP', 'SLA', 'SPR', 'SPL', 'SAR', 'SAL',
           'IRP', 'IRA', 'ILP', 'ILA', 'IPR', 'IPL', 'IAR', 'IAL',
           'ALS', 'ALI', 'ARS', 'ARI', 'ASL', 'ASR', 'AIL', 'AIR',
           'PLS', 'PLI', 'PRS', 'PRI', 'PSL', 'PSR', 'PIL', 'PIR']

def main(argv):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')

    #  subparser for checks on orientation and acquisition
    parser_check = subparsers.add_parser('checks')
    parser_check.add_argument('-f', dest='file_pattern',
                              metavar='filename_input',
                              type=str, required=True,
                              help='file or file pattern where information '
                                   'is located')
    parser_check.add_argument('-p', dest='output_path', action='store',
                              default=os.getcwd(),
                              help='output_path')
    parser_check.add_argument('-iso', dest='iso_flag', action='store_true')
    parser_check.add_argument('-ori', dest='ori_flag', action='store_true')

    #  subparser for flirtnr
    parser_flirtnr = subparsers.add_parser('flirtnr')
    parser_flirtnr.add_argument('-t_flirt', dest='flirt_transform',
                                metavar='filename_input',
                                type=str, default=None,
                                help='file with flirt transform')
    parser_flirtnr.add_argument('-ref', dest='ref_file', required=True)
    parser_flirtnr.add_argument('-flo', dest='flo_file', required=True)
    parser_flirtnr.add_argument('-t_nr', dest='nr_transform', action='store',
                                type=str, default=None,
                                help='file with nr_transform')
    parser_flirtnr.add_argument('-o', dest='output_name', action='store',
                                type=str, help='filename where to store '
                                               'transformed data', default=None)

    parser_reor = subparsers.add_parser('reor')
    parser_reor.add_argument('-new_ori', dest='final_or', choices=CHOICES,
                             required=True)
    parser_reor.add_argument('-f', dest='file', required=True)
    parser_reor.add_argument('-o', dest='output_name', action='store', type=str,
                             help='filename where to store transformed data',
                             default=None)

    parser_4to5 = subparsers.add_parser('4to5')
    parser_4to5.add_argument('-f', dest='file', required=True)
    parser_4to5.add_argument('-o', dest='output_name', action='store', type=str,
                             help='filename where to store transformed data',
                             default=None)

    parser_5to4 = subparsers.add_parser('5to4')
    parser_5to4.add_argument('-f', dest='file', required=True)
    parser_5to4.add_argument('-o', dest='output_name', action='store', type=str,
                             help='filename where to store transformed data',
                             default=None)

    parser_5to4 = subparsers.add_parser('sform')
    parser_5to4.add_argument('-f', dest='file', required=True)
    parser_5to4.add_argument('-o', dest='output_name', action='store', type=str,
                             help='filename where to store transformed data',
                             default=None)

    try:
        args = parser.parse_args(argv)

    except argparse.ArgumentTypeError:
        print('process_orientation.py flirtnr -t_flirt -o')
        sys.exit(2)

    if args.subcommand == 'checks':
        list_files = glob.glob(args.file_pattern)
        for filename in list_files:
            if args.ori_flag:
                print("%s is coronal: %s" % (filename, check_coronal(filename)))
                print("%s is axial: %s" % (filename, check_axial(filename)))
                print("%s is sagittal: %s" % (filename, check_sagittal(
                    filename)))
            if args.iso_flag:
                print("%s is isotropic: %s" % (filename, check_anisotropy(
                    filename)))

    if args.subcommand == 'flirtnr':
        if args.flirt_transform is None:
            transfo = nr_affine_to_flirt(args.ref_file, args.flo_file,
                                         args.nr_transform)
            if args.output_name is None:
                path_name = os.path.split(args.nr_transform)[0]
                name_transfo = os.path.split(args.nr_transform)[1]
                name_new = name_transfo.rstrip('.*') + '_flirt.txt'

                name_save = os.path.join(path_name, name_new)
            else:
                name_save = args.output_name
            np.savetxt(name_save, transfo)
            return
        if args.nr_transform is None:
            transfo = flirt_affine_to_nr(args.ref_file, args.flo_file,
                                         args.flirt_transform)
            if args.output_name is None:
                path_name = os.path.split(args.flirt_transform)[0]
                name_transfo = os.path.split(args.flirt_transform)[1]
                name_new = name_transfo.rstrip('.*') + '_nr.txt'

                name_save = os.path.join(path_name, name_new)
            else:
                name_save = args.output_name
            np.savetxt(name_save, transfo)
            return

    if args.subcommand == '5to4':
        new_nii = five_to_four(args.file)
        if args.output_name is None:
            name_save = args.file.rstrip('.nii.gz')+'_5to4.nii.gz'
        else:
            name_save = args.output_name
        nib.save(new_nii, name_save)

    if args.subcommand == '4to5':
        new_nii = four_to_five(args.file)
        if args.output_name is None:
            name_save = args.file.rstrip('.nii.gz')+'_4to5.nii.gz'
        else:
            name_save = args.output_name
        nib.save(new_nii, name_save)

    if args.subcommand == 'sform':
        save_sform(args.file)

    if args.subcommand == 'reor':
        file_nii = nib.load(args.file)
        init_orient = nib.orientations.io_orientation(file_nii.affine)
        init_axcodes = nib.orientations.ornt2axcodes(init_orient)
        data_new, affine_new, transfo = do_reorientation(file_nii, init_axcodes,
                                                         tuple(args.final_or))
        new_nii = nib.Nifti1Image(data_new, affine_new)
        if args.output_name is None:
            name_save = args.file.rstrip('.nii.gz')+'_'+args.final_or+'.nii.gz'
        else:
            name_save = args.output_name
        name_save_trans = name_save.rstrip('.nii.gz') + '.txt'
        nib.save(new_nii, name_save)

        np.savetxt(name_save_trans, transfo, fmt='%3.2f')

if __name__ == "__main__":
    main(sys.argv[1:])