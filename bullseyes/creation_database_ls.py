import numpy as np
import sys
import getopt
import glob
import pandas as pd
import matching_filename as mf
from difflib import SequenceMatcher
from matching_filename import match_first_degree
from creation_ls_file import (write_ls, creation_ls, read_ls_create_agglo,
                              )
from bullseye_plotting import agglo_ls_without_speclobe
import os

TYPES=['Les','Reg','Freq','Dist']
LOBES=['F','P','O','T','BG','IT']
SIDES=['L','R']
COMBINED=['BG','IT']

def split_filename(file_name):
    '''
    Operation on filename to separate path, basename and extension of a filename
    :param file_name: Filename to treat
    :return pth, fname, ext:
    '''
    pth = os.path.dirname(file_name)
    fname = os.path.basename(file_name)

    ext = None
    for special_ext in '.nii', '.nii.gz':
        ext_len = len(special_ext)
        if fname[-ext_len:].lower() == special_ext:
            ext = fname[-ext_len:]
            fname = fname[:-ext_len] if len(fname) > ext_len else ''
            break
    if ext is None:
        fname, ext = os.path.splitext(fname)
    return pth, fname, ext

def create_name_save(list_format):
    list_elements = []
    common_path = os.path.split(os.path.commonprefix(list_format))[0]
    print(common_path)
    list_common = common_path.split(os.sep)
    for l in list_format:
        split_string = l.lstrip(common_path).split(os.sep)
        for s in split_string:
            if not s in list_common and not s in list_elements:
                list_elements.append(s.replace("*",'_'))
    return common_path, '_'.join(list_elements)


def find_longest(list_seg):
    comp_s = SequenceMatcher()
    comp_s.set_seqs(list_seg[0], list_seg[-1])
    common_seg = comp_s.find_longest_match(0, len(list_seg[0]), 0,
                                           len(list_seg[-1]))
    size = common_seg.size
    for s in range(2, len(list_seg)):
        comp_s.set_seq2(list_seg[s])

        common_seg_temp = comp_s.find_longest_match(0, len(list_seg[0]), 0,
                                                    len(list_seg[-1]))
        if size > common_seg_temp.size:
            size = common_seg_temp.size
            common_seg = common_seg_temp
            print(list_seg[0][common_seg.a:common_seg.a + size])
    return common_seg


def reorder_list(list_seg, list_ref):
    new_seg = list(list_seg)
    new_ref = list(list_ref)
    common_seg = find_longest(list_seg)
    common_ref = find_longest(list_ref)
    common_seg_sub = list_seg[0][common_seg.a:common_seg.a+common_seg.size]
    common_ref_sub = list_ref[0][common_ref.a:common_ref.a + common_ref.size]
    print(common_seg_sub, common_ref_sub, "are common")
    for s in range(0,len(new_seg)):
        new_seg[s]= new_seg[s].replace(common_seg_sub, '')
    for r in range(0,len(new_ref)):
        new_ref[r]=new_ref[r].replace(common_ref_sub, '')
    common_seg = find_longest(new_seg)
    common_ref = find_longest(new_ref)
    common_seg_sub = new_seg[0][common_seg.a:common_seg.a + common_seg.size]
    common_ref_sub = new_ref[0][common_ref.a:common_ref.a + common_ref.size]
    for s in range(0,len(new_seg)):
        new_seg[s]= new_seg[s].replace(common_seg_sub, '')
    for r in range(0,len(new_ref)):
        new_ref[r]=new_ref[r].replace(common_ref_sub, '')
    print(new_ref, new_seg)
    _, _, ind_s, ind_r = match_first_degree(new_seg, new_ref)
    print(ind_s,ind_r)
    return ind_s, ind_r

def create_header_foragglo(nl=4,type=TYPES,side=SIDES,lobes=LOBES):
    header = ['ID']
    for t in TYPES:
        head_full =[t+'Tot']
        head_layers= []
        head_lobes = []
        head_lobeslayers_sides = []
        head_lobeslayers_full = []
        head_lobes_side = []
        for l in range(0, nl):
            head_layers.append(t + str(l + 1))
        for o in lobes:
            if o not in COMBINED:
                head_lobes.append(t+o)
                for l in range(0, nl):
                    head_lobeslayers_sides.append(t+o+str(l+1))
            for d in side:
                if o not in COMBINED:
                    head_lobes_side.append(t+o+d)
                    for l in range(0, nl):
                        head_lobeslayers_full.append(t+o+d+str(l+1))

        combined = ""
        for o in COMBINED:
            head_lobes_side.append(t+o)
            combined +=o
            for l in range(0, nl):
                head_lobeslayers_full.append(t+o+str(l+1))
        head_lobes.append(t+combined)
        for l in range(0, nl):
            head_lobeslayers_sides.append(t + combined + str(l + 1))
        header += head_full + head_lobeslayers_full + head_layers + \
                  head_lobes_side + head_lobeslayers_sides + head_lobes
    return header

def create_header_foragglo_corr(nl=4,type=TYPES,side=SIDES,lobes=LOBES):
    header = []
    for t in TYPES:
        head_full =[t+'Tot']
        head_layers= []
        head_lobes = []
        head_lobeslayers_sides = []
        head_lobeslayers_full = []
        head_lobes_side = []
        for l in range(0, nl):
            head_layers.append(t + str(l + 1))
        for o in lobes:
            if o not in COMBINED:
                head_lobes.append(t + o)
                for l in range(0, nl):
                    head_lobeslayers_sides.append(t + o + str(l + 1))
            for d in side:
                if o not in COMBINED:
                    head_lobes_side.append(t + o + d)
                    for l in range(0, nl):
                        head_lobeslayers_full.append(t + o + d + str(l + 1))

        combined = ""
        for o in COMBINED:
            head_lobes_side.append(t + o)
            combined += o
            for l in range(0, nl):
                head_lobeslayers_full.append(t + o + str(l + 1))
        head_lobes.append(t + combined)
        for l in range(0, nl):
            head_lobeslayers_sides.append(t + combined + str(l + 1))
        header += head_full + head_lobeslayers_full + head_layers + \
                  head_lobes_side + head_lobeslayers_sides + head_lobes
    return header


def main(argv):
    layer_pattern = None
    lobe_pattern = None
    lesion_pattern = None
    file_pattern = "LocalSummary*"
    path_result = os.getcwd()
    nl = 4
    ci = False
    try:
        opts, args = getopt.getopt(argv, "hl:o:f:s:r:ci:nl:", ["lobe=",
                                                               "layer=",
                                                               "lesion=",
                                                       "file=", "result="])
    except getopt.GetoptError:
        print('creation_ls_file.py -l <layer_file> -f <filename_write> -o '
              '<lobar_file> -s '
              '<segmentation_file> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('creation_database_ls.py -l <layer_file_pattern> -f '
                  '<filename_write_pattern> -o '
              '<lobar_file_patterm> -s '
              '<segmentation_file> -r <result_file> -nl num_layers -ci '
                  'correction_infratentorial')
            sys.exit()
        elif opt in ("-f", "--file"):
            file_pattern = arg
        elif opt in ("-l", "--layer"):
            layer_pattern = arg
        elif opt in ("-o", "--lobe"):
            lobe_pattern = arg
        elif opt in ("-s", "--lesion"):
            lesion_pattern = arg
        elif opt in ("-r", "--result"):
            result_file = arg
        elif opt in ("-p", "--path"):
            path_result = arg
        elif opt in ("-ci", "--correct_it"):
            ci = arg
        elif opt in ("-nl", "--numb_layers"):
            nl = arg

    if lesion_pattern is not None and layer_pattern is not None and \
            lobe_pattern is not None:
        lesion_list = glob.glob(lesion_pattern)
        lobe_list = glob.glob(lobe_pattern)
        layer_list = glob.glob(layer_pattern)
        ind_s, ind_o = reorder_list(lesion_list, lobe_list)
        ind_s2, ind_l = reorder_list(lesion_list, layer_list)
        lesion_fin = []
        layer_fin = []
        lobe_fin = []
        for i in range(0, len(ind_s)):
            if ind_s[i] > -1 and ind_o[i] > -1 and ind_s2[i] > -1 and ind_l[
                i]>-1:
                print(i, ind_s[i], ind_o[i])
                print(lesion_list[i], lobe_list[
                    ind_s[i]], layer_list[ind_s2[i]])
                lesion_fin.append(lesion_list[i])
                layer_fin.append(layer_list[ind_s2[i]])
                lobe_fin.append(lobe_list[ind_s[i]])
        triplet_list = list(zip(lesion_fin, lobe_fin, layer_fin))
        for (s, o, l) in triplet_list:
            vol_prob, vol_bin, vol_reg, connect = creation_ls(o,l,s)
            [path, basename, ext] = split_filename(s)
            filewrite = os.path.join(path_result, file_pattern+basename+".txt")
            write_ls(vol_prob, vol_bin, vol_reg, connect, filewrite)

    list_files = glob.glob(file_pattern)
    result_array = None
    for f in list_files:
        LesFin, RegFin, FreqFin, DistFin = read_ls_create_agglo(f,
                                                              num_layers=nl, corr_it=ci)
        LesFin2, RegFin2, FreqFin2, DistFin2 = agglo_ls_without_speclobe(
            LesFin,RegFin,4,lobe_remove=[3])
        final_array = np.concatenate((LesFin, RegFin, FreqFin,
                                      DistFin, LesFin2, RegFin2, FreqFin2,
                                      DistFin2),0).T
        [path, basename, ext] = split_filename(f)
        final_array = [basename]+ list(final_array)
        if result_array is None:
            result_array = [final_array]
        else:
            result_array = result_array + [final_array]
    header_columns = create_header_foragglo(nl)
    header_bis = create_header_foragglo_corr(lobes=['F','P','O','BG','IT'])
    header_columns_tot = header_columns + header_bis
    data_pd = pd.DataFrame(result_array, columns=header_columns_tot)
    data_pd.to_csv(result_file)


if __name__ == "__main__":
   main(sys.argv[1:])

