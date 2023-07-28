import os
import numpy as np
import pandas as pd
import math
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
    
    # subset score_df regions to atlas regions
    score_df = score_df[score_df.region_id.isin(atlas.index)]
    
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


def compute_deconvolution_n_times_nnls(mixture_replicates_path, score_type, atlas, match=True):
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


def compute_deconvolution_from_methyl_score_dir_nnls(path_to_methyl_score_dir, score_type, atlas, match=True):
    '''
    '''
    # grab all file paths in methyl_score directory
    list_methyl_score_dir = get_file_paths(path_to_methyl_score_dir)
    
    # run deconvolution on each mixture proportion
    results = []
    for path in list_methyl_score_dir:
        df = compute_deconvolution_n_times_nnls(mixture_replicates_path=path, 
                                               score_type=score_type, 
                                               atlas=atlas, 
                                               match=match)
        results.append(df)
    
    return(results)


def compute_deconvolution_naive(score_df_path, filter_cov, score_var, ref_region_df):
    
    # load score df
    score_df = pd.read_csv(score_df_path, sep='\t')

    ridxs = (score_df['number_molecules']>=filter_cov)
    fit_naive = score_df[ridxs]\
        .merge(ref_region_df[['region_id', 'ref_celltype']])\
        .groupby(['ref_celltype'])\
        [score_var].mean()\
        .reset_index()\
        .rename(columns={score_var: 'coeff'})
    fit_naive.index = fit_naive.ref_celltype
    fit_naive = fit_naive.drop(['ref_celltype'], axis=1)
    result = fit_naive.coeff
    
    return(result)


def compute_deconvolution_n_times_naive(mixture_replicates_path, filter_cov, score_var, ref_region_df):
    '''
    
    '''
    # given path to mixture grab all paths to mixture replicates
    list_mixture_dir_paths = get_file_paths(mixture_replicates_path)
    
    # run deconvolution for each replicate
    results = []
    samples_name = []
    for path in list_mixture_dir_paths:
        deconv = compute_deconvolution_naive(score_df_path=path, 
                                             filter_cov=filter_cov, 
                                             score_var=score_var,
                                            ref_region_df=ref_region_df)
        results.append(deconv)
    df = pd.concat(results, axis=1)
    
    return df 


def compute_deconvolution_from_methyl_score_dir_naive(path_to_methyl_score_dir, filter_cov, score_var, ref_region_df):
    '''
    '''
    # grab all file paths in methyl_score directory
    list_methyl_score_dir = get_file_paths(path_to_methyl_score_dir)
    
    # run deconvolution on each mixture proportion
    results = []
    
    for path in list_methyl_score_dir:
        df = compute_deconvolution_n_times_naive(mixture_replicates_path=path, 
                                               score_var=score_var,
                                               filter_cov=filter_cov,
                                               ref_region_df=ref_region_df)
        
        # rename column to match sample
        N = df.shape[1]
        new_columns = np.arange(N)
        df.columns = new_columns
        
        results.append(df)
    
    return(results)


################################
#   Functions for evaluation   #
################################

def boxplot_titration(list_of_deconvolution_dfs, cell_type, true_proportions, deconvolution_method_name):

    dfs = []

    for i in range(0, len(list_of_deconvolution_dfs)):
        df = list_of_deconvolution_dfs[i]
        phat = df[df.index == cell_type].values.squeeze()
        p_idx = np.repeat(true_proportions[i], len(phat))
        df = {'idx': p_idx, 'phat': phat}
        df = pd.DataFrame(df)
        df['idx'] = df['idx'].astype(str)
        dfs.append(df)

    df = pd.concat(dfs)

    sns.boxplot(x='idx', y='phat', data=df)

    plt.title(f'Titration Boxplots ({deconvolution_method_name})')
    plt.xlabel(f'True proportion of {cell_type}')
    plt.ylabel(f'Estimated proportion ({deconvolution_method_name})')
    plt.grid(True, alpha=0.5)
    plt.gca().set_axisbelow(True)
    
    plt.show()


def boxplot_titration_zoom(list_of_deconvolution_dfs, cell_type, true_proportions, deconvolution_method_name):

    dfs = []
    plots = []
    
    # Get dataframe into scatter plot format
    for i in range(0, len(list_of_deconvolution_dfs)):
        df = list_of_deconvolution_dfs[i]
        phat = df[df.index == cell_type].values.squeeze()
        p_idx = np.repeat(true_proportions[i], len(phat))
        df = {'idx': p_idx, 'phat': phat}
        df = pd.DataFrame(df)
        df['idx'] = df['idx'].astype(str)
        dfs.append(df)

    # Calculate the grid size: square root of the number of dataframes
    grid_size = math.ceil(math.sqrt(len(dfs)))

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Flatten the axs array for easy iterating
    axs = axs.ravel()

    # Create a boxplot on each subplot using your data
    for i, df in enumerate(dfs):
        sns.boxplot(data=df, x="idx", y="phat", ax=axs[i], zorder=2)
        plot_name = true_proportions[i]
        axs[i].set_title(f"True proportion: {plot_name}")  # Set individual titles for subplots
        axs[i].set_xlabel(f'True proportion of {cell_type}') 
        axs[i].set_ylabel(f'Estimated proportion ({deconvolution_method_name})') 
        
    # If there are more subplots than dataframes, remove the extras
    if len(dfs) < len(axs):
        for i in range(len(dfs), len(axs)):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    