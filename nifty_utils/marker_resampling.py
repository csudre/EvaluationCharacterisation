import numpy as np
import nibabel as nib
import os
import sys
import argparse
from nifty_utils.orientation import read_matrix

def get_list_orig_indices(filename):
    nii_img = nib.load(filename)
    data = nii_img.get_data()
    indices = np.asarray(np.where(data>0))
    return indices, nii_img.affine

def transform_indices_new_space(list_ind, orig_tomm, new_tomm, transfo):
    ind_orig_final = np.concatenate([list_ind, np.ones([1,
        list_ind.shape[1]])],0)
    inv_transfo = np.linalg.inv(transfo)

    new_tovox = np.linalg.inv(new_tomm)
    orig_mm = np.matmul(orig_tomm, ind_orig_final)
    transfo_mm = np.matmul(inv_transfo, orig_mm)
    new_ind = np.matmul(new_tovox, transfo_mm)

    #new_ind = np.matmul(inv_transfo, ind_orig_final)
    return np.asarray(new_ind.T, dtype=int)

def fill_new_image(filename, list_ind):
    basic_nii = nib.load(filename)
    results = np.zeros_like(basic_nii.get_data())
    for i in range(0, list_ind.shape[0]):
        results[list_ind[i,0], list_ind[i,1], list_ind[i,2]] = 1
    results_nii = nib.Nifti1Image(results, basic_nii.affine)
    return results_nii

def process_transfo_markers(init_seg, final_img, transfo_file):
    matrix_transfo = read_matrix(transfo_file)
    list_ind, affine_floating = get_list_orig_indices(init_seg)
    nii_target = nib.load(final_img)

    trans_ind = transform_indices_new_space(list_ind, affine_floating,
                                            nii_target.affine, matrix_transfo)
    new_img = fill_new_image(final_img, trans_ind)
    return new_img

def main(argv):
    parser = argparse.ArgumentParser(description='Create resampling of '
                                                 'markers')
    parser.add_argument('-trans', dest='transfo',
                        metavar='affine transformation',
                        type=str, required=True,
                        help='affine transformation file')
    parser.add_argument('-seg', dest='seg', action='store',
                        required=True, help='path to seg')
    parser.add_argument('-ref', dest='ref', action='store',required=True,
                        help='path to reference')

    parser.add_argument('-name_save', dest='name_save', action='store',
                        default='',
                        help='name of output file')
    parser.add_argument('-o', dest='output_path', action='store',
                        help='where to save results', default=os.getcwd())

    try:
        args = parser.parse_args()
        # print(args.accumulate(args.integers))

    except argparse.ArgumentTypeError:
        print('')
        sys.exit(2)

    new_img = process_transfo_markers(args.seg, args.ref, args.transfo)
    final_save = os.path.join(args.output_path, args.name_save)
    nib.save(new_img, final_save)



if __name__ == "__main__":
    main(sys.argv[1:])