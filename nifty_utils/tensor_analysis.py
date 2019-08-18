import nibabel as nib
import scipy.linalg as la
import scipy.stats as st
import os
import argparse
import sys
import numpy as np


def create_tensor_from_4d(filename):
    nii = nib.load(filename)
    data = nii.get_data()
    data_new = np.zeros(np.concatenate([np.shape(data)[0:3], [9]]))
    data_new[..., 0:3] = data[..., 0:3]
    data_new[..., 3] = data[..., 1]
    data_new[..., 4:6] = data[..., 3:5]
    data_new[..., 6] = data[..., 2]
    data_new[..., 7] = data[..., 4]
    data_new[..., -1] = data[..., -1]
    data_tensor = np.expand_dims(data_new, -1)
    data_tensor = np.reshape(data_tensor, np.concatenate([np.shape(data)[
                                                          0:3], [3, 3]]))
    return data_tensor, nii.affine


def svd_tensor(tensor_image, filemask):
    nii_mask = nib.load(filemask)
    mask = nii_mask.get_data()
    eigen_vectors = np.zeros_like(tensor_image)
    eigen_values = np.zeros(np.shape(tensor_image)[0:-1])
    test_evec = np.zeros_like(tensor_image)
    test_eval = np.zeros(np.shape(tensor_image)[0:-1])
    data_shape = mask.shape
    for i in range(0, data_shape[0]):
        print("Processing slice %i" % i)
        for j in range(0, data_shape[1]):
            for k in range(0, data_shape[2]):
                if mask[i, j, k]:
                    temp_val, temp_vec = la.eigh(1000 *
                        tensor_image[i, j, k, ...])
                    final_val = np.sort(np.abs(temp_val)) / 1000.0

                    sort = np.argsort(np.abs(temp_val))
                    sign_val = np.sign(temp_val)
                    mul_sign = np.tile(sign_val, [3, 1])
                    new_vec = mul_sign * temp_vec / 1000.0
                    final_vec = new_vec[:, sort]
                    eigen_values[i, j, k, ...], eigen_vectors[i, j, k,
                                                :] = final_val, final_vec

                    test_eval[i, j, k, ...], test_evec[i, j, k, ...] = la.eigh(1000.0 * tensor_image[i,j,
                                                                         k,
                                                                         ...])
    return eigen_values, eigen_vectors


def extract_from_tensor(eigen_values):
    mean_val = np.mean(eigen_values, axis=-1)
    second_moment = st.moment(eigen_values, 2, axis=-1)
    third_moment = st.moment(eigen_values, 3, axis=-1)
    r1 = np.sqrt(np.sum(np.square(eigen_values), -1))
    fa = np.sqrt(1.5 * np.sum(np.square(eigen_values - np.expand_dims(
        mean_val,-1)),-1))/r1

    fa2 = np.sqrt(3 * second_moment / (2*np.square(mean_val) +
                                      second_moment))
    md = mean_val
    mo = np.sqrt(2) * third_moment / np.power(second_moment, 3/2)
    return fa, md, mo


def analyse_full_tensor_save(file_tensor, file_mask):
    tensor, affine = create_tensor_from_4d(file_tensor)
    path_tensor = os.path.split(file_tensor)[0]
    val, vec = svd_tensor(tensor, file_mask)
    fa, md, mo = extract_from_tensor(val)
    fa_nii = nib.Nifti1Image(fa, affine)
    md_nii = nib.Nifti1Image(md, affine)
    mo_nii = nib.Nifti1Image(mo, affine)

    l1_nii = nib.Nifti1Image(val[..., 2], affine)
    l2_nii = nib.Nifti1Image(val[..., 1], affine)
    l3_nii = nib.Nifti1Image(val[..., 0], affine)
    v1_nii = nib.Nifti1Image(vec[..., 2], affine)
    v2_nii = nib.Nifti1Image(vec[..., 1], affine)
    v3_nii = nib.Nifti1Image(vec[..., 0], affine)

    name_fa = os.path.join(path_tensor, 'dtifit_FA.nii.gz')
    name_md = os.path.join(path_tensor, 'dtifit_MD.nii.gz')
    name_mo = os.path.join(path_tensor, 'dtifit_MO.nii.gz')
    name_l1 = os.path.join(path_tensor, 'dtifit_L1.nii.gz')
    name_l2 = os.path.join(path_tensor, 'dtifit_L2.nii.gz')
    name_l3 = os.path.join(path_tensor, 'dtifit_L3.nii.gz')
    name_v1 = os.path.join(path_tensor, 'dtifit_V1.nii.gz')
    name_v2 = os.path.join(path_tensor, 'dtifit_V2.nii.gz')
    name_v3 = os.path.join(path_tensor, 'dtifit_V3.nii.gz')

    nib.save(fa_nii, name_fa)
    nib.save(md_nii, name_md)
    nib.save(mo_nii, name_mo)
    nib.save(l1_nii, name_l1)
    nib.save(l2_nii, name_l2)
    nib.save(l3_nii, name_l3)
    nib.save(v1_nii, name_v1)
    nib.save(v2_nii, name_v2)
    nib.save(v3_nii, name_v3)




# tensor = create_tensor_from_4d(
#     '/Users/csudre/Documents/Tractography/17226225_01/17226225_01_38_results'
#     '/DTI_17226225_01_38_cleaned_tensors.nii.gz')

path_dwi = '/Users/csudre/Documents/Tractography/17226225_01/17226225_01_38_results'
mask = os.path.join(path_dwi, 'DTI_17226225_01_38_cleaned_mask.nii.gz')
dwi = os.path.join(path_dwi, 'DTI_17226225_01_38_cleaned_tensors.nii.gz')
analyse_full_tensor_save(dwi, mask)
print('finished')


