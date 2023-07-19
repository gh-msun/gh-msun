import os
import numpy as np
import pandas as pd
import random
import glob
import datetime
import itertools
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import nbformat

from scipy.optimize import nnls


def get_file_paths(directory):

    list_paths = []

    for filename in os.listdir(directory):
        list_paths.append(os.path.abspath(os.path.join(directory, filename)))

    return(list_paths)


def compute_deconvolution_nnls(score_df_path, score_type, atlas, match=True):
    '''
    Run nonnegative least squares ||Ax-b||_2. 
    The solution x is the deconvolution of b.
    
    Reasoning for match=True:
    Note that for lower total read count for a mixture, there be regions that
    are missing in score_df. At 1M reads, this not a problem.
    
    score_df -- methylation score dataframe
    score_type -- hypo or hyper score: e.g. 'frac_alpha_leq_25pct'
    atlas -- atlas dataframe
    '''
    # load score df
    score_df = pd.read_csv(score_df_path, sep='\t')
    score_df.index = score_df.region_id
    
    b = score_df[score_type]
    A = atlas
    
    # match index between A and b
    if match:
        region_count_before = A.shape[0]
        A = A[A.index.isin(b.index)]
        region_count_after = A.shape[0]
        region_count_diff = region_count_before - region_count_after 
        print(f'Dropped: {region_count_diff} regions.')
    
    # sort the indices for A to match b indices
    A_sorted = A.loc[b.index, :]
    
    # run NNLS
    fit = nnls(A_sorted, b)
    x = pd.Series(fit[0], index=A_sorted.columns)
    
    return(x)


def compute_deconvolution_n_times(mixture_replicates_path, score_type, atlas, match=True):
    '''
    
    mixture_replicates_path -- path to a mixture (proportion) directory of replicates (e.g. ../E1B_E18CD4_E18CD8_E18NK_E18MONO_E18NEUTRO/)
    output: pandas df
    
    '''
    # given path to mixture grab all paths to mixture replicates
    list_mixture_dir_paths = get_file_paths(mixture_replicates_path)
    
    # run deconvolution for each replicate
    results = []
    samples_name = []
    for path in list_mixture_dir_paths:
        deconv = compute_deconvolution_nnls(score_df_path=path, 
                                           atlas=atlas, 
                                           score_type=score_type, 
                                           match=match)
        results.append(deconv)
    df = pd.concat(results, axis=1)
    
    return df 


def compute_deconvolution_methyl_score_dir(path_to_methyl_score_dir, score_type, atlas, match=True):
    '''
    '''
    # grab all file paths in methyl_score directory
    list_methyl_score_dir = get_file_paths(path_to_methyl_score_dir)
    
    # run deconvolution on each mixture proportion
    results = []
    for path in list_methyl_score_dir:
        df = compute_deconvolution_n_times(mixture_replicates_path=path, 
                                               score_type=score_type, 
                                               atlas=atlas, 
                                               match=match)
        results.append(df)
    
    return(results)