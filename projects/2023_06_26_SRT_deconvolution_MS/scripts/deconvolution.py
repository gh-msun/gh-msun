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
import cvxpy as cp
from sklearn.linear_model import HuberRegressor

from scipy.optimize import nnls


def get_file_paths(directory):

    list_paths = []

    for filename in os.listdir(directory):
        list_paths.append(os.path.abspath(os.path.join(directory, filename)))

    return(list_paths)


######################################
#              NNLS                  #
######################################

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
     #   print(f'Dropped: {region_count_diff} regions.')
    
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


######################################
#              Naive                 #
######################################

def compute_deconvolution_naive(score_df_path, filter_cov, score_var, ref_region_df):
    
    
    # score_df itself should be input
    
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


####################################################
#              Huber (force zero)                  #
####################################################

def compute_deconvolution_huber(score_df_path, score_type, atlas, epsilon, match=True, force_zero=False):
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
       # print(f'Dropped: {region_count_diff} regions.')
    
    # sort the indices for A to match b indices
    A_sorted = A.loc[b.index, :]
    
    # alpha = 0 no regularization
    huber = HuberRegressor(alpha=0.0, 
                           epsilon=epsilon, 
                           fit_intercept = False)
    huber.fit(A_sorted, b)
    
    # run huber
    x = pd.Series(huber.coef_, index=A_sorted.columns) 
    
    # if force_zero is set to True then set negative to 0
    if force_zero:
        x[x < 0] = 0
    
    return(x)


def compute_deconvolution_n_times_huber(mixture_replicates_path, score_type, atlas, epsilon, match=True, force_zero=False):
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
        deconv = compute_deconvolution_huber(score_df_path=path, 
                                           atlas=atlas, 
                                           score_type=score_type, 
                                           match=match,
                                           epsilon=epsilon,
                                           force_zero=force_zero)
        results.append(deconv)
    df = pd.concat(results, axis=1)
    
    return df 


def compute_deconvolution_from_methyl_score_dir_huber(path_to_methyl_score_dir, score_type, atlas, epsilon, match=True, force_zero=False):
    '''
    '''
    # grab all file paths in methyl_score directory
    list_methyl_score_dir = get_file_paths(path_to_methyl_score_dir)
    
    # run deconvolution on each mixture proportion
    results = []
    for path in list_methyl_score_dir:
        df = compute_deconvolution_n_times_huber(mixture_replicates_path=path, 
                                               score_type=score_type, 
                                               atlas=atlas, 
                                               epsilon=epsilon,
                                               match=match,
                                               force_zero=force_zero)
        results.append(df)
    
    return(results)


####################################################
#              Huber (nonzero constraint)          #
####################################################

def compute_deconvolution_nnhuber(score_df_path, score_type, atlas, epsilon, match=True):
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
    def nnHuber(A, b, delta):
        
        n = A.shape[1]
        x = cp.Variable(n)

        loss = A @ x - b
        huber_loss = cp.sum(cp.huber(loss, delta))

        # Objective
        objective = cp.Minimize(huber_loss)

        # Nonnegativity constraints
        constraints = [x >= 0]

        # Problem
        prob = cp.Problem(objective, constraints)

        # Solve
        prob.solve()

        return(x.value)
    
    
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
       # print(f'Dropped: {region_count_diff} regions.')
    
    # sort the indices for A to match b indices
    A_sorted = A.loc[b.index, :]
    
    # run huber
    x = nnHuber(A_sorted.to_numpy(), b, delta=epsilon)
    x = pd.Series(x, index=A_sorted.columns)
    
    # extremely small, less than -10^10, just round to 0
    threshold = 10 ** -10
    abs(x) < threshold
    x[abs(x) < threshold] = 0
    
    return(x)


def compute_deconvolution_n_times_nnhuber(mixture_replicates_path, score_type, atlas, epsilon, match=True):
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
        deconv = compute_deconvolution_nnhuber(score_df_path=path, 
                                           atlas=atlas, 
                                           score_type=score_type, 
                                           epsilon=epsilon,
                                           match=match)
        results.append(deconv)
    df = pd.concat(results, axis=1)
    
    return df 


def compute_deconvolution_from_methyl_score_dir_nnhuber(path_to_methyl_score_dir, score_type, atlas, epsilon, match=True):
    '''
    '''
    # grab all file paths in methyl_score directory
    list_methyl_score_dir = get_file_paths(path_to_methyl_score_dir)
    
    # run deconvolution on each mixture proportion
    results = []
    for path in list_methyl_score_dir:
        df = compute_deconvolution_n_times_nnhuber(mixture_replicates_path=path, 
                                               score_type=score_type, 
                                               atlas=atlas, 
                                               epsilon=epsilon,
                                               match=match)
        results.append(df)
    
    return(results)


########################################
#  Plotting for titration experiments  #
########################################

def boxplot_titration_combined(list_of_deconvolution_dfs_naive, list_of_deconvolution_dfs_nnls, cell_type, true_proportions):

    dfs = []
    M = len(list_of_deconvolution_dfs_naive)

    for i in range(0, M):
        df_naive = list_of_deconvolution_dfs_naive[i]
        df_nnls = list_of_deconvolution_dfs_nnls[i]
        
        phat_naive = df_naive[df_naive.index == cell_type].values.squeeze()
        phat_nnls = df_nnls[df_nnls.index == cell_type].values.squeeze()
          
        # get method name
        str_naive = ['naive'] * len(phat_naive)
        str_nnls = ['nnls'] * len(phat_nnls)
        
        p_idx_naive = np.repeat(true_proportions[i], len(phat_naive))
        p_idx_nnls = np.repeat(true_proportions[i], len(phat_nnls))
        
        df_naive = {'idx': p_idx_naive, 'phat': phat_naive, 'method':str_naive}
        df_nnls = {'idx': p_idx_nnls, 'phat': phat_nnls, 'method':str_nnls}
        df_naive = pd.DataFrame(df_naive)
        df_nnls = pd.DataFrame(df_nnls)
        df = pd.concat([df_naive, df_nnls])
        df = pd.DataFrame(df)
        df['idx'] = df['idx'].astype(str)
        dfs.append(df)

    df = pd.concat(dfs)
    
    plt.figure(figsize=(12, 8))  # width and height in inches

    # # Draw horizontal lines
    # for y_value in true_proportions:
    #     plt.axhline(y=y_value, color='black', linestyle='--',linewidth=0.5)

    custom_palette = {'naive': 'red', 'nnls': 'blue'}
    sns.violinplot(x='idx', y='phat', data=df, hue='method', palette=custom_palette, linewidth=0)
    
    # sns.stripplot(x='idx', y='phat', data=df, hue='method', dodge=True, jitter=False, size=10, edgecolor=['red', 'blue'], linewidth=1, palette=['red', 'blue'])
    
    # # Change the outline color
    # for i, art in enumerate(violin_plot.patches):
    #     # print(i, art)
    #     # edge_color = 'red' if i == 0 else 'blue'
    #     edge_color='red'
    #     art.set_edgecolor(edge_color)

    plt.title(f'Titration Boxplots')
    plt.xlabel(f'True proportion of {cell_type}')
    plt.ylabel(f'Estimated proportion')
    plt.grid(True, alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.xticks(range(len(true_proportions)), true_proportions, rotation='vertical')

    plt.show()

    
def boxplot_titration_zoom_combined(list_of_deconvolution_dfs_naive, list_of_deconvolution_dfs_nnls, cell_type, true_proportions):

    dfs = []
    plots = []
    M = len(list_of_deconvolution_dfs_naive)

    # Get dataframe into scatter plot format
    for i in range(0, M):
        df_naive = list_of_deconvolution_dfs_naive[i]
        df_nnls = list_of_deconvolution_dfs_nnls[i]
        
        phat_naive = df_naive[df_naive.index == cell_type].values.squeeze()
        phat_nnls = df_nnls[df_nnls.index == cell_type].values.squeeze()
          
        # get method name
        str_naive = ['naive'] * len(phat_naive)
        str_nnls = ['nnls'] * len(phat_nnls)
        
        p_idx_naive = np.repeat(true_proportions[i], len(phat_naive))
        p_idx_nnls = np.repeat(true_proportions[i], len(phat_nnls))
        
        df_naive = {'idx': p_idx_naive, 'phat': phat_naive, 'method':str_naive}
        df_nnls = {'idx': p_idx_nnls, 'phat': phat_nnls, 'method':str_nnls}
        df_naive = pd.DataFrame(df_naive)
        df_nnls = pd.DataFrame(df_nnls)
        df = pd.concat([df_naive, df_nnls])
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
    custom_palette = {'naive': 'red', 'nnls': 'blue'}
    for i, df in enumerate(dfs):
        sns.violinplot(data=df, x="idx", y="phat", ax=axs[i], zorder=2, hue='method', palette=custom_palette)
        
        plot_name = true_proportions[i]
        axs[i].set_xlabel('') 
        axs[i].set_ylabel('') 
        axs[i].legend().remove()
        axs[i].axhline(y= plot_name, color='magenta', linestyle='--',linewidth=2)
        
        for label in axs[i].get_xticklabels():
            label.set_fontweight('bold')
    

    # If there are more subplots than dataframes, remove the extras
    if len(dfs) < len(axs):
        for i in range(len(dfs), len(axs)):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    

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
    
    plt.figure(figsize=(12, 6))  # width and height in inches

    sns.boxplot(x='idx', y='phat', data=df, color='dodgerblue')
    
    # add ground truth dash
    df_true = {'idx': true_proportions, 'y': true_proportions}
    df_true = pd.DataFrame(df_true)
    df_true['idx'] = df_true['idx'].astype(str)
    plt.scatter(x='idx', y='y', data=df_true, color='magenta', marker='_', s=150)
    
   # sns.violinplot(x='idx', y='phat', data=df, linewidth=0, color='blue')

    plt.title(f'Titration Boxplots ({deconvolution_method_name})')
    plt.xlabel(f'True proportion of {cell_type}')
    plt.ylabel(f'Estimated proportion ({deconvolution_method_name})')
    plt.grid(True, alpha=0.5)
    plt.gca().set_axisbelow(True)
    
    plt.xticks(range(len(true_proportions)), true_proportions, rotation='vertical')
    
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
        sns.boxplot(data=df, x="idx", y="phat", ax=axs[i], zorder=2, color='dodgerblue')
        #sns.violinplot(data=df, x="idx", y="phat", ax=axs[i], zorder=2, color='dodgerblue')
        plot_name = true_proportions[i]
        axs[i].set_xlabel('') 
        axs[i].set_ylabel('')
        axs[i].axhline(y= plot_name, color='magenta', linestyle='-',linewidth=2)
        
        for label in axs[i].get_xticklabels():
            # label.set_fontweight('bold')
            label.set_color('magenta')

    # If there are more subplots than dataframes, remove the extras
    if len(dfs) < len(axs):
        for i in range(len(dfs), len(axs)):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
    
    
###################
#   LOD 95 Plot   #
###################

def lod95_detect_plot(names, titrating_celltypes, titrating_celltype_proportion, deconvolution_preds, detection_threshold, lod=0.95):
    # celltype atlas, deconvolution_predictions list(s), titration list, titrating cell type proportions, decision threshold
    # celltype atlas, deconvolution list(s), titration list, titrating cell type proportions, +/- tolerance
    # support 1 vs all
    # titrating_celltypes = ['immune_t_effector', 'immune_treg']
    # titrating_celltype_proportion = TITRATION_LIST
    # deconvolution_preds = deconvolution_nnls
    # detection_threshold =  0.005
    # lod = 0.95

    data = []
    lod95_value = []

    # <- start loop
    for i, celltype in enumerate(titrating_celltypes):
        
        # celltype_idx = titrating_celltypes.index(celltype)
        preds_for_celltype = deconvolution_preds[i]
        
        print(i)
        print(celltype)

        p_detect = []

        for df in preds_for_celltype:
            m, n = df.shape
            phat = df[df.index == celltype]
            phat_boolean = phat > detection_threshold
            p_detected = sum(phat_boolean.values.squeeze()) / n
            p_detect.append(p_detected)

        ###### ----- plotting

        # Replace these with your data
        x_values = titrating_celltype_proportion
        y_values = p_detect 

        data.append((x_values, y_values))

        # Sorting the x_values and y_values in the order of x_values
        sorted_indices = np.argsort(x_values)
        x_values_sorted = np.array(x_values)[sorted_indices]
        y_values_sorted = np.array(y_values)[sorted_indices]

        if x_values_sorted[0] == 0:
            y_values_sorted[0] = 0

        # Interpolating to find the x-value
        y_target = lod
        x_target = np.interp(y_target, y_values_sorted, x_values_sorted)
        lod95_value.append(x_target)


    for i, (x_values, y_values) in enumerate(data):
        name = names[i]
        lod95 = round(lod95_value[i], 5)
        label = f'{name}: {lod95}'
        if x_values[-1] == 0:
            y_values[-1] = 0
        sns.scatterplot(x=x_values, y=y_values, label=label)
        sns.lineplot(x=x_values, y=y_values)

    # Set the limits and spacing of the x and y axes
    # plt.xticks(ticks=[i/10 for i in range(11)])
    # plt.yticks(ticks=[i/10 for i in range(11)])
    plt.xlim(0, 0.1)
    plt.ylim(0, 1)

    plt.axhline(y=y_target, xmin=0, xmax=x_target+1, color='blue', linestyle='--', linewidth=0.8)

    plt.grid(True, alpha=0.5)
    plt.gca().set_axisbelow(True)

    plt.xlabel('Titration Proportion')
    plt.ylabel('% Correctly Identified (x/20)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    
    
def lod95_within_proportion_plot(names, titrating_celltypes, titrating_celltype_proportion, deconvolution_preds, tolerance=0.3, lod=0.95):
    # celltype atlas, deconvolution_predictions list(s), titration list, titrating cell type proportions, decision threshold
    # celltype atlas, deconvolution list(s), titration list, titrating cell type proportions, +/- tolerance
    # support 1 vs all
    # titrating_celltypes = ['immune_t_effector', 'immune_treg']
    # titrating_celltype_proportion = TITRATION_LIST
    # deconvolution_preds = deconvolution_nnls
    # detection_threshold =  0.005
    # lod = 0.95

    data = []
    lod95_value = []

    # <- start loop
    for i, celltype in enumerate(titrating_celltypes):
        
      #  celltype_idx = titrating_celltypes.index(celltype)
        preds_for_celltype = deconvolution_preds[i]

        p_correct = []

        for i in range(0,len(preds_for_celltype)):
            df = preds_for_celltype[i]
            m, n = df.shape
            eps = titrating_celltype_proportion[i] * tolerance
            true_p = titrating_celltype_proportion[i]
            true_p_min = true_p - eps 
            true_p_max = true_p + eps
            phat = df[df.index == celltype]
            phat = phat.values.squeeze()
            phat_boolean = [(true_p_min <= x <= true_p_max) for x in phat]
            p_correct.append(sum(phat_boolean) / n)

        ###### ----- plotting

        # Replace these with your data
        x_values = titrating_celltype_proportion
        y_values = p_correct

        data.append((x_values, y_values))

        # Sorting the x_values and y_values in the order of x_values
        sorted_indices = np.argsort(x_values)
        x_values_sorted = np.array(x_values)[sorted_indices]
        y_values_sorted = np.array(y_values)[sorted_indices]

        if x_values_sorted[0] == 0:
            y_values_sorted[0] = 0

        # Interpolating to find the x-value
        y_target = lod
        x_target = np.interp(y_target, y_values_sorted, x_values_sorted)
        lod95_value.append(x_target)

    for i, (x_values, y_values) in enumerate(data):
        name = names[i]
        lod95 = round(lod95_value[i], 5)
        label = f'{name}: {lod95}'
        if x_values[-1] == 0:
            y_values[-1] = 0
        sns.scatterplot(x=x_values, y=y_values, label=label)
        sns.lineplot(x=x_values, y=y_values)

    # Set the limits and spacing of the x and y axes
    # plt.xticks(ticks=[i/10 for i in range(11)])
    # plt.yticks(ticks=[i/10 for i in range(11)])
    plt.xlim(0, 0.1)
    plt.ylim(0, 1)

    plt.axhline(y=y_target, xmin=0, xmax=x_target+1, color='blue', linestyle='--', linewidth=0.8)

    plt.grid(True, alpha=0.5)
    plt.gca().set_axisbelow(True)

    plt.xlabel('Titration Proportion')
    plt.ylabel('% Correctly Identified (x/20)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    

    
    
    
# def boxplot_titration_log(list_of_deconvolution_dfs, cell_type, true_proportions, deconvolution_method_name, eps):
    
#     # convert to log scale
#     deconvolution_naive_log = []
#     true_proportions = list(np.log(np.array(true_proportions)))
#     for i in range(0,len(list_of_deconvolution_dfs)):
#         log_transformed = np.log(list_of_deconvolution_dfs[i])
#         deconvolution_naive_log.append(log_transformed)
    
#     dfs = []
#     for i in range(0, len(deconvolution_naive_log)):
#         df = deconvolution_naive_log[i]
#         phat = df[df.index == cell_type].values.squeeze()
#         p_idx = np.repeat(true_proportions[i], len(phat))
#         df = {'idx': p_idx, 'phat': phat}
#         df = pd.DataFrame(df)
#         df['idx'] = df['idx'].astype(str)
#         dfs.append(df)
        
#     df = pd.concat(dfs)
    
#     plt.figure(figsize=(12, 8))  # width and height in inches

#   #   sns.boxplot(x='idx', y='phat', data=df)
#     sns.violinplot(x='idx', y='phat', data=df)

#     plt.title(f'Titration Boxplots ({deconvolution_method_name})')
#     plt.xlabel(f'True proportion of {cell_type}')
#     plt.ylabel(f'Estimated proportion ({deconvolution_method_name})')
#     plt.grid(True, alpha=0.5)
#     plt.gca().set_axisbelow(True)
    
#     true_proportions = list(np.round(np.array(true_proportions), 1))
#     plt.xticks(range(len(true_proportions)), true_proportions, rotation='vertical')
#     plt.show()

    
    
# def boxplot_background_zoom_all(list_of_deconvolution_dfs, cell_type, true_proportions, deconvolution_method_name):

#     dfs = [] 

#     for i in range(0, len(list_of_deconvolution_dfs)):
#         df_ = list_of_deconvolution_dfs[i]
#         df_t = df_.transpose()
#         df_t.columns = ['B', 'CD4', 'CD8', 'NK', 'Mono', 'Eosi', 'Neutro', 'Eryth', 'Mega', 'Eryth-P']
#         df_melt = pd.melt(df_t, value_vars=['B', 'CD4', 'CD8', 'NK', 'Mono', 'Eosi', 'Neutro', 'Eryth', 'Mega', 'Eryth-P'])
#         n = df_melt.shape[0]
#         idx = np.repeat(i, n)
#         df_melt['idx'] = true_proportions[i]
#         dfs.append(df_melt)

#     # Calculate the grid size: square root of the number of dataframes
#     grid_size = math.ceil(math.sqrt(len(dfs)))

#     # Create a figure with a grid of subplots
#     fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

#     # Flatten the axs array for easy iterating
#     axs = axs.ravel()
    
#     # Create a boxplot on each subplot using your data
#     for i, df in enumerate(dfs):
#         sns.boxplot(data=df, x="idx", y="value", hue='variable', ax=axs[i], zorder=2)
#         plot_name = true_proportions[i]
#         axs[i].set_xlabel('') 
#         axs[i].set_ylabel('') 
        
#         # Hide the legend
#         if i != len(axs) - 1:
#             axs[i].get_legend().remove()

#     # If there are more subplots than dataframes, remove the extras
#     if len(dfs) < len(axs):
#         for i in range(len(dfs), len(axs)):
#             fig.delaxes(axs[i])    
            
#     # Set legend to the last subplot
#     handles, labels = axs[-1].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

#     plt.tight_layout()
#     plt.show()
    

# def background_estimates_boxplot(list_of_deconvolution_dfs, cell_types, titration_proportions, method):

#     dfs = [] 

#     for i in range(0, len(list_of_deconvolution_dfs)):

#         df_ = list_of_deconvolution_dfs[i]
#         df_t = df_.transpose()
#         df_t.columns = cell_types
#         df_melt = pd.melt(df_t, value_vars=cell_types)

#         n = df_melt.shape[0]
#         idx = np.repeat(i, n)
#         df_melt['idx'] = idx

#         dfs.append(df_melt)

#     df_bplot = pd.concat(dfs, axis=0)
    
#     plt.figure(figsize=(16, 12)) 
    
#     sns.set_style("whitegrid")
#     sns.boxplot(x="idx", y="value",
#                 hue="variable",
#                 data=df_bplot)
    
#     plt.xticks(range(len(titration_proportions)), titration_proportions, rotation='vertical')
#     plt.xlabel('Titration proportions')
#     plt.ylabel(f'Estimated proportions ({method})')
#     plt.legend(bbox_to_anchor=(0.5,-0.2), loc="upper center", borderaxespad=0, ncol=5)
#     plt.tight_layout()
#     plt.show()

    
# def background_estimates_boxplot_zoom(proportion, list_of_deconvolution_dfs, cell_types, titration_proportions, method):
    
#     proportion_idx = titration_proportions.index(proportion)

#     df_ = list_of_deconvolution_dfs[proportion_idx]
#     df_t = df_.transpose()
#     df_t.columns = cell_types
#     df_melt = pd.melt(df_t, value_vars=cell_types)

#     n = df_melt.shape[0]
#     idx = np.repeat(proportion_idx, n)
#     df_melt['idx'] = idx

#     plt.figure(figsize=(16, 12)) 
    
#     sns.set_style("whitegrid")
#     sns.boxplot(x="idx", y="value",
#                 hue="variable",
#                 data=df_melt)
    
#     plt.xlabel('Titration proportions')
#     plt.ylabel(f'Estimated proportions ({method})')
#     plt.legend(bbox_to_anchor=(0.5,-0.2), loc="upper center", borderaxespad=0, ncol=5)
#     plt.tight_layout()
#     plt.show()
    
    