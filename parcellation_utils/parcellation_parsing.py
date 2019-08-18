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
import argparse


def get_dict_parc(l):
    tree = ET.parse(l)
    labels = tree.findall('.//labels')[0].findall('.//number')
    vols = tree.findall('.//labels')[0].findall('.//volumeProb')
    dict_labels = {la.text:v.text for la, v in zip(labels, vols)}
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
    HEADER = ['Label']
    dict1 = {}
    with open(association_file,
              "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            if not row[1] in HEADER:
                dict1[row[1]] = {key: key+'_'+value for (key, value) in zip(
                                          header[2:], row[2:])}
        infile.close()
    return dict1
