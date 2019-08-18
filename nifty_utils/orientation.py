import logging
import nibabel as nib
import numpy as np

CHOICES = ['LAS', 'LAI', 'LPS', 'LPI', 'LSA', 'LSP', 'LIA', 'LIP',
           'RAS', 'RAI', 'RPS', 'RPI', 'RSA', 'RSP', 'RIA', 'RIP',
           'SRP', 'SRA', 'SLP', 'SLA', 'SPR', 'SPL', 'SAR', 'SAL',
           'IRP', 'IRA', 'ILP', 'ILA', 'IPR', 'IPL', 'IAR', 'IAL',
           'ALS', 'ALI', 'ARS', 'ARI', 'ASL', 'ASR', 'AIL', 'AIR',
           'PLS', 'PLI', 'PRS', 'PRI', 'PSL', 'PSR', 'PIL', 'PIR']


def check_anisotropy(filename):
    """
    Check if the associated filename is an isotropic image
    :param filename: filename to assess
    :return:
    """
    img = nib.load(filename)
    img_pixdim = img.header.get_zooms()
    pix_rat = np.asarray([1.0 * img_pixdim[0] / img_pixdim[1],
                          1.0 * img_pixdim[1] / img_pixdim[2],
                          1.0 * img_pixdim[0] / img_pixdim[2]])
    abs_rat = np.abs(pix_rat - 1)

    return bool(any(ar > 0.2 for ar in abs_rat))


def check_axial(filename):
    """
    Check if the associated image is acquired in axial acquisition
    :param filename:
    :return:
    """
    img = nib.load(filename)
    img_pixdim = img.get_header().get_zooms()
    axcodes = nib.orientations.aff2axcodes(img.affine)
    if 'A' in axcodes:
        # print ("S here")
        pix_ap = img_pixdim[axcodes.index('A')]
    else:
        pix_ap = img_pixdim[axcodes.index('P')]
    if 'L' in axcodes:
            # print ("S here")
        pix_lr = img_pixdim[axcodes.index('L')]
    else:
        pix_lr = img_pixdim[axcodes.index('R')]
    return bool(pix_ap == pix_lr)


def check_coronal(filename):
    """
    Check if the associated image is acquired in coronal acquisition
    :param filename:
    :return:
    """
    img = nib.load(filename)
    img_pixdim = img.get_header().get_zooms()
    axcodes = nib.orientations.aff2axcodes(img.affine)
    if 'L' in axcodes:
        # print ("S here")
        pix_lr = img_pixdim[axcodes.index('L')]
    else:
        pix_lr = img_pixdim[axcodes.index('R')]
    if 'I' in axcodes:
            # print ("S here")
        pix_is = img_pixdim[axcodes.index('I')]
    else:
        pix_is = img_pixdim[axcodes.index('S')]
    return bool(pix_is == pix_lr)



def check_sagittal(filename):
    """
    Check if the associated image is acquired in sagittal acquisition
    :param filename:
    :return:
    """
    img = nib.load(filename)
    img_pixdim = img.get_header().get_zooms()
    axcodes = nib.orientations.aff2axcodes(img.affine)
    if 'A' in axcodes:
        # print ("S here")
        pix_ap = img_pixdim[axcodes.index('A')]
    else:
        pix_ap = img_pixdim[axcodes.index('P')]
    if 'I' in axcodes:
            # print ("S here")
        pix_is = img_pixdim[axcodes.index('I')]
    else:
        pix_is = img_pixdim[axcodes.index('S')]
    return bool(pix_is == pix_ap)


def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    logger = logging.getLogger('compute_orientation')
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)
    # if np.any(np.isnan(ornt_init)) or np.any(np.isnan(ornt_fin)):
    #     raise ValueError:
    #         "unknown axcodes %s, %s", ornt_init, ornt_fin

    try:
        ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
        return ornt_transf, ornt_init, ornt_fin
    except ValueError:
        logger.error('reorientation transform error: %s, %s', ornt_init,
                     ornt_fin)


def do_reorientation(nii_image, init_axcodes, final_axcodes):
    """
    Performs the reorientation (changing order of axes)

    :param nii_image: nib image object
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    logger = logging.getLogger('reorientation')
    ornt_transf, ornt_init, ornt_fin = \
        compute_orientation(init_axcodes, final_axcodes)
    data_array = nii_image.get_data()
    affine = nii_image.affine
    test = nib.orientations.inv_ornt_aff(ornt_transf, data_array.shape)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array, affine, test
    try:
        return nib.orientations.apply_orientation(data_array, ornt_transf), \
               np.matmul(affine, test), test
    except ValueError:
        logger.error('reorientation undecided %s, %s', ornt_init, ornt_fin)


def flirt_affine_to_nr(ref_file, flo_file, flirt_aff):
    """
    Transform a flirt affine matrix of transformation into the niftyreg
    equivalent
    :param ref_file: reference file
    :param flo_file: floating image
    :param flirt_aff: affine flirt file
    :return: niftyreg affine matrix
    """
    ref_nii = nib.load(ref_file)
    ref_matrix = None
    flo_matrix = None
    if isinstance(ref_nii, nib.Nifti1Image):
        if ref_nii.header['sform_code'] > 0:
            ref_matrix = ref_nii.get_sform()
        else:
            ref_matrix = ref_nii.get_qform()
    flo_nii = nib.load(flo_file)
    if isinstance(flo_nii, nib.Nifti1Image):
        if flo_nii.header['sform_code'] > 0:
            flo_matrix = flo_nii.get_sform()
        else:
            flo_matrix = flo_nii.get_qform()

    if ref_matrix is None:
        ref_matrix = np.eye(4)
    if flo_matrix is None:
        flo_matrix = np.eye(4)

    norm_ref = np.sqrt(np.sum(np.square(ref_matrix[0:3, 0:3]), 1))
    norm_flo = np.sqrt(np.sum(np.square(flo_matrix[0:3, 0:3]), 1))
    abs_ref = np.diag(np.concatenate((norm_ref, [1])))
    abs_flo = np.diag(np.concatenate((norm_flo, [1])))
    inv_abs_flo = np.linalg.inv(abs_flo)
    mat_flirt = read_matrix(flirt_aff)
    inv_mat_flirt = np.linalg.inv(mat_flirt)
    mat = np.matmul(inv_abs_flo, inv_mat_flirt)
    mat = np.matmul(mat, abs_ref)
    mat = np.matmul(flo_matrix, mat)
    inv_ref = np.linalg.inv(ref_matrix)
    nr_aff = np.matmul(mat, inv_ref)
    print(nr_aff)
    return nr_aff


def nr_affine_to_flirt(ref_file, flo_file, nr_aff):
    """
    Transform niftyreg affine matrix into flirt affine matrix
    :param ref_file: reference file
    :param flo_file: floating file
    :param nr_aff: nifty reg affine matrix file
    :return: flirt affine matrix
    """
    ref_nii = nib.load(ref_file)
    if ref_nii.header['sform_code'] > 0:
        ref_matrix = ref_nii.get_sform()
    else:
        ref_matrix = ref_nii.get_qform()
    flo_nii = nib.load(flo_file)
    if flo_nii.header['sform_code'] > 0:
        flo_matrix = flo_nii.get_sform()
    else:
        flo_matrix = flo_nii.get_qform()

    norm_ref = np.sqrt(np.sum(np.square(ref_matrix[0:3, 0:3]), 1))
    norm_flo = np.sqrt(np.sum(np.square(flo_matrix[0:3, 0:3]), 1))

    abs_ref = np.diag(np.concatenate((norm_ref, [1])))
    abs_flo = np.diag(np.concatenate((norm_flo, [1])))

    inv_abs_ref = np.linalg.inv(abs_ref)
    inv_flo = np.linalg.inv(flo_matrix)

    mat_nr = read_matrix(nr_aff)
    print("nr aff is ", mat_nr)
    print("ref matrix is ", ref_matrix)
    mat = np.matmul(mat_nr, ref_matrix)

    mat = np.matmul(mat, inv_abs_ref)
    mat = np.matmul(inv_flo, mat)
    mat = np.matmul(abs_flo, mat)
    flirt_aff = np.linalg.inv(mat)
    print(flirt_aff)
    return flirt_aff


def read_matrix(transfo_file):
    """
    Read the affine matrix from file
    :param transfo_file: transformation text file
    :return: 4x4 affine matrix as np array
    """
    from numpy import loadtxt
    lines = loadtxt(transfo_file)
    return np.asarray(lines)





