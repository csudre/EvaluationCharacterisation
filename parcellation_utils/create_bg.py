import nibabel as nib
import numpy as np
import glob
import sys, getopt, os



def check_anisotropy(filename):
    img = nib.load(filename)

    img_pixdim = img.get_header().get_zooms()
    # print(img_pixdim)
    if any(pixdim > 1.5 for pixdim in img_pixdim):
        return True
    else:
        return False

def check_axial(filename):
    img = nib.load(filename)
    img_pixdim = img.get_header().get_zooms()
    axcodes = nib.orientations.aff2axcodes(img.affine)
    if 'S' in axcodes:
        # print ("S here")
        if img_pixdim[axcodes.index('S')] > 2:
            return True
        else:
            return False
    elif 'I' in axcodes:
        # print ("I here")
        if img_pixdim[axcodes.index('I')] > 2:
            return True
        else:
            return False
    return False

def check_coronal(filename):
    img = nib.load(filename)
    img_pixdim = img.get_header().get_zooms()
    axcodes = nib.orientations.aff2axcodes(img.affine)
    if 'A' in axcodes:
        # print ("S here")
        if img_pixdim[axcodes.index('A')] > 2:
            return True
        else:
            return False
    elif 'P' in axcodes:
        # print ("I here")
        if img_pixdim[axcodes.index('P')] > 2:
            return True
        else:
            return False
    return False

def create_hemisphere(filename):
    left_array = (31,33,    38,40,42,    44,    49,    51,    53,    55,
                 57,
             59,    61,    63,    65,    67,    76,    89,    90,    91,    92,    93,    94,    97,    102,    104,    106,    108,    110,    114,    116,   118,    120,   122,    124,    126,    130,    134,    136,    138,    140,    142,    144,    146,    148,    150,    152,    154,    156,    158,    162,    164,    166,    168,    170,    172,    174,    176,    178,    180,    182,    184,    186,    188,    192,    194,    196,    198,    200,    202,    204,    206,208)
    right_array = (24,32,    37,    39,    41,    43,    48,    50,    52,
                   54,    56,    58,    60,    62,    64,    66,    77,    81,    82,    83,    84,    85,    86,    96,    101,    103,    105,    107,    109,    113,    115,117,    119,    121,    123,    125,129,    133 ,   135,    137,    139,    141,    143,    145,    147,    \
     149,    151,    153,    155,    157,    161,    163,    165,    167,    169,    171,    173,    175,    177,    179,    181,    183,    185,    187,    191,    193,    195,    197,199,    201,    203,    205, 207)
    img = nib.load(filename)
    img_data = img.get_data()
    right_hemi = np.zeros_like(img_data)
    left_hemi = np.zeros_like(img_data)
    for v_r in right_array:
        right_hemi[img_data == v_r] = 1
        right_nii = nib.Nifti1Image(right_hemi, img.affine)
    for v_l in left_array:
        left_hemi[img_data == v_l] = 1
        left_nii = nib.Nifti1Image(left_hemi, img.affine)
    return right_nii, left_nii

def create_bg(filename):
    bg_array = (24,	31,	32,	33,	37,	38,	56,	57,	58,	59,	60,	61,	76,	77)
    img = nib.load(filename)
    img_data = img.get_data()
    bg_seg = np.zeros_like(img_data)
    for v_r in bg_array:
        bg_seg[img_data == v_r] = 1
    bg_nii = nib.Nifti1Image(bg_seg, img.affine)
    return bg_nii

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <regexp> -n <name> ')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            rule = arg
    print(rule)
    list_files = glob.glob(rule)
    print(len(list_files))
    for filename in list_files:
        path = os.path.split(filename)[0]
        name = os.path.split(filename)[1]
        array_name = name.split('_')
        name_fin = array_name[1] + '_' + array_name[2]
        bg_nii = create_bg(filename)
        nib.save(bg_nii, os.path.join(path,'DGM_%s.nii.gz' % name_fin))
        # nib.save(left_nii, os.path.join(path, 'LeftHemi_%s.nii.gz') %name_fin)
        # flag = False
        # flag = check_anisotropy(filename)
        # flag_cor = check_coronal(filename)
        # if flag_cor is True:
        #     name = os.path.split(filename)[1]
        #     print(name.rstrip('.nii.gz').lstrip('T1Gad_'))


if __name__ == "__main__":
   main(sys.argv[1:])