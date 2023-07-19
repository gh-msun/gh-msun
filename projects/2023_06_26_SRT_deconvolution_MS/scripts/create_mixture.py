import pandas as pd
import glob
import numpy as np
import itertools
import functools
import os
import regex as re
import random

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.types import IntegerType, LongType, ArrayType, StringType, DoubleType
from pyspark.sql.functions import udf, explode, broadcast, count, lit, length, col
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType


def load_parquet_dataframe(parquet_path, cell_types, spark, verbose=True):
    '''
    Load parquet file by cell type and count the number of rows.
    Takes in parquet files where the samples have been collapsed by cell type.
    
    Arguments:
    parquet_path -- path to the directory with source cell type reads to mix from
    cell_types -- list of cell type to load for mixing
    '''
    
    # Load the parquet files for selected cell types & count rows
    parquet_df = []
    total_reads_per_celltype = []
    
    if verbose: print('>>> Load parquet files and count rows... <<<')
    for cell_type in cell_types:
        if verbose: print(f'----------> Loading cell type: {cell_type}')
        df = spark.read.parquet(f'{parquet_path}collapsed_reads_{cell_type}/')
        parquet_df.append(df)
        total_reads_per_celltype.append(df.count())

    total_reads_per_celltype = np.array(total_reads_per_celltype)
    
    if verbose: print('>>> Complete. <<< \n')

    return(parquet_df, total_reads_per_celltype)
    

def one_to_many_seeds(seed, n):
    '''
    Generate seeds (between 0 and 1 million)
    '''
    random.seed(seed)
    seeds = [random.randint(0, 10**6) for _ in range(n)]
    return(seeds)


def generate_mixture_dir_name_string(list_celltype_name, list_proportion):
    '''
    Generate name for cell type given list of cell type name and proportions
    E = '0.'
    e.g. ['B', 'CD4', 'CD8', 'NK', 'Mono', 'Neutro'] and np.array([0.05, 0.015, 0.5, 0.6, 0.7, 0]).
    Output string: 'E05B_E015CD4_E5CD8_E6NK_E7MONO_ENEUTRO'
    '''
    
    # Replace '0.' with 'E', add corresponding letter from list_a, and join all elements with '_'
    dir_name = '_'.join(['E' + str(b)[2:] + a.upper() for a, b in zip(list_celltype_name, list_proportion)])
    return(dir_name)


def reverse_translate(input_string):
    pass


def mix_celltypes(parquet_df, total_reads_per_celltype, cell_types, total_reads_to_sample, proportions, seed, result_path, spark, verbose, save=False, itr=None):
    ''' Mix reads from different cell types based on given proportion and total reads to sample.
    Note: Data is loaded once in mix_celltypes_n_times() to avoid loading dataframes repeatedly.
    
    Arguments:
    paquet_df -- list of dataframes loaded in mix_celltypes_n_times()
    total_reads_per_celltype -- calculated while reading in dataframe (nrow of each df)
    cell_types -- list of cell type to mix
    total_reads_to_sample -- integer representing the total number of reads to sample across all cell types
    proportions -- list of proportions to sample for each cell type
    seed -- seed for .sample()
    result_path -- path to output parquet file (e.g. experiment/mixture/mix_50B_50CD4/)
    itr -- mixture iteration for creating multiple mixtures (for file naming)
    
    Output:
    mixture -- pyspark.sql.dataframe.DataFrame
    '''
    
    if verbose: print(f'--> seed: {seed}')
    
    # compute fraction to sample for each cell type (later convert to index)
    n_reads_to_sample = proportions * total_reads_to_sample
    sampling_fraction = n_reads_to_sample / total_reads_per_celltype
    if verbose: print(f'Sampling fraction: {sampling_fraction}')
    
    # sample reads from each cell type
    sampled_df = []
    
    if verbose: print('--> Sample rows for each cell type...')
    for i in range(0, len(cell_types)):
        if verbose: print(f'----------> Sampling cell type: {cell_types[i]}')
        df = parquet_df[i]
        frac = sampling_fraction[i]
        df_sample = df.sample(False, frac, seed)
        sampled_df.append(df_sample)
        n_sampled = df_sample.count()
        if verbose: print(f'----------> {n_sampled}')
    
    # combine reads
    if verbose: print('--> Combining sampled reads into one dataframe...')
    mixture = functools.reduce(DataFrame.union, sampled_df)
    
    if save:
        # create file name 
        seed_string = str(int(seed))
        celltype_string = '_'.join(cell_types)
        proportion_str = [str(i) for i in proportions]
        proportion_string = '_'.join(proportion_str)
        mixture_itr = f'mix{itr}_'
        file_name = mixture_itr + \
                    f'seed{seed_string}' + \
                    '.parquet/'

        if verbose: print('--> Saving parquet file...')
        save_path = result_path + file_name
        mixture.write.mode('overwrite').parquet(save_path)
        if verbose: print(f'--> Saved to: {save_path}')
    
    return(mixture)


def mix_celltypes_n_times(parquet_df, total_reads_per_celltype, n, cell_types, cell_type_abridged_name, total_reads_to_sample, proportions, seed, result_path, spark, verbose, save=False):
    '''Create n mixtures by mixing reads from different cell types based on given proportion and total reads to sample. 
    Loads the parquet files.
    Calls mix_celltypes() n times.
    
    Arguments:
    n -- total number of mixtures to make
    cell_types -- list of cell type to mix
    cell_type_abridged_name -- abridged name of cell types to be used for naming output directory
    total_reads_to_sample -- integer representing the total number of reads to sample across all cell types
    proportions -- list of proportions to sample for each cell type
    seed -- seed for .sample()
    result_path -- path to the mixture directory where all the mixtures for each proportion list will be saved (e.g. experiment/mixture/)
    '''
    
    # Create output directory
    dir_name = generate_mixture_dir_name_string(cell_type_abridged_name, proportions)
    dir_name = dir_name + '/' #+ '_seed' + str(seed) + '/'
    dir_path = result_path + dir_name
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    # generate seeds
    seeds = one_to_many_seeds(seed, n=n)
    
    # Create n mixtures   
    for i in range(0, n):
        
        print(f'----------> Creating mixture {i}... ')
        mixture = mix_celltypes(parquet_df=parquet_df,
                               total_reads_per_celltype=total_reads_per_celltype,
                               cell_types=cell_types,
                               total_reads_to_sample=total_reads_to_sample, 
                               proportions=proportions, 
                               seed=seeds[i],
                               result_path=dir_path,
                               spark=spark,
                               save=save,
                               itr=i,
                               verbose=verbose)


def mix_celltypes_multiple_proportions(parquet_df, total_reads_per_celltype, n, cell_types, cell_type_abridged_name, total_reads_to_sample, list_of_proportions, seed, result_path, spark, verbose=False, save=False):
    '''Create n mixtures by mixing reads from different cell types based on given proportion and total reads to sample. 
    Calls mix_celltypes_n_times()  times.
    
    Arguments:
    n -- total number of replicate mixtures to make per proportion
    cell_types -- list of cell type to mix
    cell_type_abridged_name -- abridged name of cell types to be used for naming output directory
    total_reads_to_sample -- integer representing the total number of reads to sample across all cell types
    list_of_proportions -- list(list()) list of list of proportions to sample for each cell type
    seed -- seed for .sample()
    result_path -- path to the mixture directory where all the mixtures for each proportion list will be saved (e.g. experiment/mixture/)
    '''
    
    print('>>> Start mixing... <<<')
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    seeds = one_to_many_seeds(seed, n=n)
    i = 0

    for proportion in list_of_proportions:

        print(f"--> PROPORTION: {proportion}")
        mix_celltypes_n_times(
                     parquet_df=parquet_df,
                     total_reads_per_celltype=total_reads_per_celltype,
                     n=n,
                     cell_types=cell_types,
                     cell_type_abridged_name=cell_type_abridged_name,
                     total_reads_to_sample=total_reads_to_sample, 
                     proportions=proportion, 
                     seed=seeds[i],
                     result_path=result_path,
                     spark=spark,
                     save=save,
                     verbose=verbose)
        i += 1
    
    print(">>> Complete. <<< \n")
    

    
    
    
# # UPDATE HOME!
# os.environ["SPARK_HOME"] = "/home/ec2-user/mambaforge/envs/2023_06_26_SRT_deconvolution_MS/lib/python3.7/site-packages/pyspark"
# # THIS needs to be set-up before running the notebook
# os.environ["SPARK_LOCAL_DIRS"] = "/temp"
# os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# spark_conf = SparkConf()
# spark_conf.set("spark.ui.showConsoleProgress", "True")
# spark_conf.set("spark.executor.instances", "2")
# spark_conf.set("spark.executor.cores", "2")
# spark_conf.set("spark.executor.memory", "16g")
# spark_conf.set("spark.driver.memory", "64g")
# spark_conf.set("spark.driver.maxResultSize", "32g")
# spark_conf.set("spark.parquet.filterPushdown", "true")
# spark_conf.set("spark.local.dir", "/temp")
# spark_conf.getAll()

# sc = SparkContext(conf=spark_conf)
# sc.setLogLevel("ERROR")
# spark = SparkSession(sc)
