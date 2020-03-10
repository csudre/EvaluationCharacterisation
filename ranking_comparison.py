import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, rankdata
import sys
import argparse
import os
import getopt
from PyNonpar.twosample_paired import paired_ranks_test as paired_nonpar

GREATER = ['dice', 'ppv', 'jaccard', 'sensitivity', 'specificity', 'accuracy',
           'tpr', 'ltpr']
LESS = ['hausdorff', 'adist', 'vol_diff', 'fpr', 'fnr', 'lfpr', 'lfnr']


def pairwise_comp_wilcox(array, direction='greater'):
    """
    function defining the p-values associated to each one-sided paired non
    parameteric comparison
    :param array: input for a given measure of all the results (N x M) where
    N is the number of subjects and M the number of methods
    :param direction: "two_sided, greater or less" according to the test
    direction
    :return: array (M x M) indicated the p-value associated to the
    test comparing  method i (row) to
    method j (column) according to direction
    """
    results = np.eye(array.shape[1])
    # results2 = np.eye(array.shape[1])
    for i in range(0, array.shape[1]):
        for j in range(i+1, array.shape[1]):
            if array.shape[0] < 20:
                stats1, p1 = wilcoxon(array[:, i]-array[:, j],
                                      alternative=direction)
                stats2, p2 = wilcoxon(array[:, j] - array[:, i],
                                      alternative=direction)
            else:
                p1 = paired_nonpar(list(array[:,i]), list(array[:,j]),
                                   alternative=direction, var_equal=False,
                                   quantile="t")[-1]
                p2 = paired_nonpar(list(array[:, j]), list(array[:, i]),
                                   alternative=direction,
                                   var_equal=False, quantile="t")[-1]
            results[i, j] = p1
            results[j, i] = p2
    return results


def rank_significance(array, direction='greater', p_thresh=0.05):
    """
    Derive the associated ranking based on the counting of the number of
    methods that perform significantly worse
    :param array: input for a given measure of all the results (N x M) where
    N is the number of subjects and M the number of methods
    :param direction: "two_sided, greater or less" according to the test
    direction
    :param p_thresh: significance threshold
    :return: vector indicating the rank of each method
    """
    results_wilcox = pairwise_comp_wilcox(array, direction)
    bin_wilcox = results_wilcox < p_thresh
    count_worse = np.sum(bin_wilcox, 1)
    rank_final = len(count_worse) - rankdata(count_worse) + 1
    return rank_final


def main(argv):

    parser = argparse.ArgumentParser(description='Ranking  procedure')
    parser.add_argument('-f', dest='file_in', metavar='file with the database of results',
                        type=str, required=True,
                        help='File to read the data from')
    parser.add_argument('-t', dest='thresh', default=0.05,
                        type=float)
    parser.add_argument('-id', dest='id_indic', type=str, help='indicator '
                                                               'used for subject id', default='id')
    parser.add_argument('-team', dest='team_indic', type=str, help='indicator used for team id', default='method')

    parser.add_argument('-o', dest='output_file', action='store',
                        help='output file', type=str)

    try:
        args = parser.parse_args(argv)

    except getopt.GetoptError:
        print('ranking_comparison.py -id <field of the csv for subject '
              'id> -team <field in csv for team id> -o <output file> -t '
              '<significance '
              'threshold> ')
        sys.exit(2)

    if not os.path.exists(args.file_in):
        ValueError("No file to load!!!")
    df_results = pd.read_csv(args.file_in)
    list_teams = np.unique(df_results[args.team_indic])
    list_df_team = []
    dict_results = {}
    dict_results['sum'] = np.zeros([len(list_teams)])
    for t in list_teams:
        list_df_team.append(df_results[df_results[args.team_indic]==t])
    list_columns = df_results.columns
    count = 0
    lower_columns = [c.lower() for c in list_columns]
    for (c_l, c_i) in zip(lower_columns,list_columns):
        if c_l in GREATER:
            print(c_i)
            array_temp = [list_df_team[i][c_i] for i in range(0,
                                                              len(list_teams))]
            rank_temp = rank_significance(np.vstack(array_temp).T,
                                          direction='greater',
                                          p_thresh=args.thresh)
            dict_results[c_i] = rank_temp
            dict_results['sum'] += rank_temp
            count += 1
        if c_l in LESS:
            print(c_i)
            array_temp = [list_df_team[i][c_i] for i in range(0,
                                                              len(list_teams))]
            rank_temp = rank_significance(np.vstack(array_temp).T,
                                          direction='less',
                                          p_thresh=args.thresh)
            dict_results[c_i] = rank_temp
            dict_results['sum'] += rank_temp
            count += 1
    dict_results['sum'] /= count
    pd_results = pd.DataFrame.from_dict(dict_results)
    pd_results['team'] = list_teams
    pd_results.to_csv(args.output_file)

if __name__ == "__main__":
    main(sys.argv[1:])



