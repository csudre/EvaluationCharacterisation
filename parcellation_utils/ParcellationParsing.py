from xml.etree import ElementTree as ET
import csv
import glob
import sys
import getopt
import numpy as np
import ast
import nibabel as nib
import os
import pandas as pd

def get_dict_parc(l):
    tree = ET.parse(l)
    labels = tree.findall('.//labels')[0].findall('.//number')
    vols = tree.findall('.//labels')[0].findall('.//volumeProb')
    dict_labels = {la.text:v.text for la,v in zip(labels,vols)}
    return dict_labels

def get_dict_match(dict_labels, dict_hierarchies):
    dict_new = {}
    for (keys, values) in zip(dict_hierarchies.keys(),
                              dict_hierarchies.values()):
        volume = float(dict_labels[keys])
        for values_bis in values.values():
            if values_bis in dict_new.keys():
                dict_new[values_bis] += float(volume)
            else:
                dict_new[values_bis] = float(volume)
    return dict_new

def get_hierarchy(association_file):
    HEADER=['Label']
    dict1={}
    with open(association_file,
              "rb") as infile:
        reader = csv.reader(infile)
        header = reader.next()
        for row in reader:
            if not row[1] in HEADER:
                dict1[row[1]] = {key: key+'_'+value for (key, value) in zip(
                                          header[2:],row[ 2:])}
        infile.close()
    return dict1

def main(argv):
    association_file = '/Users/csudre/Documents/GIFHierarchy.csv'
    path_final = os.path.split(association_file)[0]
    filename = 'ParcellationsHierarchy.csv'
    demographic_file = None
    pattern = "*.xml"
    exclusion = "zzzzzzzz"
    strip_name_right = '_NeuroMorph.xml'
    strip_name_left = ''

    try:
        opts, args = getopt.getopt(argv, "ha:p:f:e:d:r:l:t:", ["assoc=",
                                                            "pattern=",
                                                     "out_file=", "exclude=",
                                                         "demographic=",
                                                             "left=",
                                                             "right=", "path="])
    except getopt.GetoptError:
        print('ParcellationParsing.py -a <association_file> -p <pattern> -f '
              'output_file -e '
              '<exclusion_pattern> -d <demographic_file> -l <left_strip> -r '
              '<right_strip> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <regexp> -m <regexp> ')
            sys.exit()
        elif opt in ("-a", "--assoc"):
            association_file = arg
        elif opt in ("-p", "--pattern"):
            pattern = arg
        elif opt in ("-f", "--out_file"):
            filename = arg
        elif opt in ("-e", "--exclude"):
            exclusion = arg
        elif opt in ("-d", "--demographic"):
            demographic_file = arg
        elif opt in ("-r", "--right"):
            strip_name_right = arg
        elif opt in ("-l", "--left"):
            strip_name_left = arg
        elif opt in ("-t", "--path"):
            path_final = arg
    if demographic_file is not None:
        demographic_df = pd.DataFrame.from_csv(path=demographic_file)
        demographic_dict = demographic_df.to_dict()
    else:
        demographic_dict = {}
    path_results = os.path.join(path_final, filename)
    dict_hierarchy = get_hierarchy(association_file)
    list_parcellation = glob.glob(pattern)
    test = get_dict_parc(list_parcellation[0])
    dict_new = get_dict_match(test, dict_hierarchy)
    list_keys_columns = dict_new.keys()
    list_keys_columns.sort()
    columns = ['ID'] + demographic_dict.keys() + ['TIV'] + list_keys_columns
    dict_total = {c:[] for c in columns}
    for l in list_parcellation:
        name = os.path.split(l)[1].rstrip(strip_name_right)
        name = name.lstrip(strip_name_left)
        print(name)
        if 'DOB' in demographic_dict.keys():
            if exclusion not in l and name in demographic_dict['DOB'].keys():
                d=get_dict_parc(l)
                dict_fin = get_dict_match(d, dict_hierarchy)
                dict_fin['File'] = l
                TIV = 0
                for c in list_keys_columns:
                    if '6_' in c and c not in ('6_0', '6_1', '6_2', '6_3',
                                               '6_4'):
                        TIV += float(dict_fin[c])
                    dict_total[c].append(dict_fin[c])
                dict_total['TIV'].append(TIV)
                if name in demographic_dict['DOB'].keys():
                    dict_total['ID'].append(name)
                    for k in demographic_dict.keys():
                        if k == 'sex':
                            dict_total[k].append(demographic_dict[k][name]-1)
                        else:
                            dict_total[k].append(demographic_dict[k][name])
        else:
            d = get_dict_parc(l)
            dict_fin = get_dict_match(d, dict_hierarchy)
            dict_fin['File'] = l
            dict_total['ID'].append(name)
            TIV = 0
            for c in list_keys_columns:
                if '6_' in c and c not in ('6_0', '6_1','6_2','6_3'):
                    TIV += float(dict_fin[c])
                dict_total[c].append(dict_fin[c])
            dict_total['TIV'].append(TIV)

    df = pd.DataFrame(dict_total)
    df.to_csv(path_results, header=True, columns=columns)

if __name__ == "__main__":
   main(sys.argv[1:])