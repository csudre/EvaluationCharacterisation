import nibabel as nib
import os
import sys
import numpy as np
import argparse
DICT_CPT = {'A': 'P','P': 'A', 'R': 'L', 'L': 'R', 'S': 'I', 'I': 'S' }

def choose_u_lambda_fromorisplit(u_vect, lamb_use, orientation=('L','A','S'),
                                                                splitdir='A'):
    if splitdir in orientation:
        index = orientation.index(splitdir)
    else:
        index = orientation.index(DICT_CPT[splitdir])

    amax = np.argmax(np.abs(np.asarray(u_vect)), 1)
    if index in amax:
        use_index = list(amax).index(index)
    return u_vect[use_index], lamb_use[use_index]


def create_split(file, label, split_numb=2, splitdir='A'):
    parc = nib.load(file)
    data = parc.get_data()
    seg = np.where(data == label, np.ones_like(data), np.zeros_like(data))
    indices = np.asarray(np.where(data == label)).T
    demeaned_ind = indices - np.tile(np.expand_dims(np.mean(indices, 0), 0),
                                     [np.shape(indices)[0], 1])
    u_vect, lamb_svd, _ = np.linalg.svd(demeaned_ind.T)
    orientation_parc = nib.orientations.ornt2axcodes(
        nib.orientations.io_orientation(
        parc.affine))
    u_use, lamb_use = choose_u_lambda_fromorisplit(u_vect, lamb_svd,
                                                   orientation=orientation_parc,
                                                   splitdir=splitdir)

    print(orientation_parc)
    print(u_vect[0])

    com_ind = np.mean(indices, 0)
    p_cut = []
    if split_numb % 2 == 0:
        p_cut = [com_ind]
        for i in range(0, split_numb-2):
            p_cut.append(com_ind + i * np.sqrt(lamb_use)/split_numb *
                         u_use)
            p_cut.append(com_ind - i * np.sqrt(lamb_use)/split_numb *
                         u_use)
    else:
        for i in range(0, split_numb-2):
            p_cut.append(com_ind + i * np.sqrt(lamb_use) / split_numb *
                                               u_use +
                         np.sqrt(lamb_svd[0]) / (2 * split_numb) * u_use)
            p_cut.append(com_ind - i * np.sqrt(lamb_use) / split_numb *
                                               u_use -
                         np.sqrt(lamb_use) / (2 * split_numb) * u_use)

    sign_dot = []
    for p in p_cut:
        if splitdir in orientation_parc:
            vect_ind = p - indices
        else:
            vect_ind = indices - p
        dot_prod = np.matmul(vect_ind, u_use)
        sign_dot.append(np.sign(dot_prod))
    sign_full = np.asarray(sign_dot)
    sign_full = np.where(sign_full < 0, np.zeros_like(sign_full), sign_full)
    final_class = np.sum(sign_full.T, 1)

    split_image = np.zeros_like(data)
    if np.min(final_class) == 0:
        final_class += 1
    for i in range(0, np.size(final_class)):
        split_image[indices[i, 0],
                    indices[i, 1], indices[i, 2]] = final_class[i]
    split_nii = nib.Nifti1Image(split_image, parc.affine)
    return split_nii


def main(argv):

    parser = argparse.ArgumentParser(description='Transform GIF parcellation '
                                                 'into FS parcellation.')
    parser.add_argument('-p', dest='parcellation', metavar='filename_input',
                        type=str, required=True,
                        help='file where the input parcellation is located')

    parser.add_argument('-o', dest='output_path', action='store',
                        default=os.getcwd(),
                        help='output_path')
    parser.add_argument('-id', dest='id_subject', action='store', type=str,
                        help='subject_name', default='pid')
    parser.add_argument('-l', dest='label', action='store', type=int,
                        help='label to split', default=87)
    parser.add_argument('-n', dest='numb_split', action='store', type=int,
                        default=2, help='number of split to operate')
    parser.add_argument('-splitdir', dest='splitdir', action='store',
                        type=str, choices=['L','R','A','P','S','I'],
                        help='direction in which to privilege the split',
                        default='A')
    try:
        args = parser.parse_args(argv)
        # print(args.accumulate(args.integers))

    except argparse.ArgumentTypeError:
        print('BrainHearts.py -f <filename_database> -g <grouping> -d '
              '<dependent variable> -i <independent variables>')
        print('The list of independent variables must always start with the '
              'Age')
        sys.exit(2)

    new_split = create_split(args.parcellation, args.label, args.numb_split,
                             args.splitdir)
    name_save = os.path.join(args.output_path, 'Split' + '_' + str(
                                 args.numb_split)+'_' +
                             str(args.label)+'_' + args.id_subject+'.nii.gz')
    nib.save(new_split, name_save)
    print("All done for split!")


if __name__ == "__main__":
    main(sys.argv[1:])





