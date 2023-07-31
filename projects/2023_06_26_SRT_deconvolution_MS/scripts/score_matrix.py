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


def get_file_paths(directory):

    list_paths = []

    for filename in os.listdir(directory):
        list_paths.append(os.path.abspath(os.path.join(directory, filename)))

    return(list_paths)


def compute_frag_scores(cpg_number_cutoff: int, schema, kmers, rates_leq, rates_geq) -> pd.DataFrame:
    
    """
    Function that returns a function, used for reduce
    """
    
    def compute_frag_scores_inner(pat_df: pd.DataFrame) -> pd.DataFrame:
        
        data = pat_df.copy()
        data['offset_min'] = (data['region_cpg_index_min'] - data['cpg_index_min']).clip(lower=0)
        data['offset_max'] = np.minimum(
            data['region_cpg_index_max'] - data['cpg_index_min'], 
            data['cpg_index_max'] - data['cpg_index_min'])
        data['trimmed_pat'] = data.apply(lambda x: x['pat_string'][x['offset_min']:(x['offset_max']+1)], axis=1)
        #--- Filter molecules based on observed CpG loci
        observed_cpg_number = (data['trimmed_pat'].str.count('C')+data['trimmed_pat'].str.count('T'))
        ridxs = (observed_cpg_number>=cpg_number_cutoff)
        data = data[ridxs].copy()
        if (data.shape[0]>0):
            # Compute k-mer methylation states
            for k in kmers:
                data['meth_k%i'%k] = data['trimmed_pat']\
                    .apply(lambda x: len(re.findall('[C]{%i}'%k, x, overlapped=True)))
                data['unmeth_k%i'%k] = data['trimmed_pat']\
                    .apply(lambda x: len(re.findall('[T]{%i}'%k, x, overlapped=True)))
                data['total_k%i'%k] = data['trimmed_pat']\
                    .apply(lambda x: len(re.findall('[TC]{%i}'%k, x, overlapped=True)))
            # Compute alpha distribution metrics
            data['alpha'] = data['meth_k1']/data['total_k1']
            for rate in rates_leq:
                data['frac_alpha_leq_%ipct'%(100*rate)] = np.where(data['alpha']<=rate, 1, 0)
            for rate in rates_geq:
                data['frac_alpha_geq_%ipct'%(100*rate)] = np.where(data['alpha']>=rate, 1, 0)
            # Expand entries that correspond to multiple molecules
            data['number_molecules'] = data['number_molecules'].apply(lambda x: list(range(x)))
            data = data.explode('number_molecules')
            data['number_molecules'] = 1
            # Aggregate metrics
            #rv = data.groupby(['region_id', 'sample_id'])\
            rv = data.groupby(['region_id'])\
                [['meth_k1', 'unmeth_k1', 'total_k1',
                  'meth_k3', 'unmeth_k3', 'total_k3',
                  'meth_k4', 'unmeth_k4', 'total_k4',
                  'frac_alpha_leq_25pct', 'frac_alpha_geq_75pct', 'number_molecules']].sum()\
                .reset_index()
            rv['frac_alpha_leq_25pct'] = rv['frac_alpha_leq_25pct']/rv['number_molecules']
            rv['frac_alpha_geq_75pct'] = rv['frac_alpha_geq_75pct']/rv['number_molecules']
        else:
            rv = pd.DataFrame(columns=schema.names)
                      
        
        return rv[schema.names]

    return compute_frag_scores_inner

        
def score_matrix(parquet_path, result_path, pat_cols, region_df, batch_size, schema, spark, compute_frag_scores_udf, save=False, verbose=False):
    '''
    Function to compute fragment score from one parquet file: 1 parquet file --> 1 score matrix.
    '''
    # Load single parquet file
    pat_df = spark.read.parquet(parquet_path).select(*pat_cols)
    
    scores_df = pat_df\
            .groupby('region_id')\
            .applyInPandas(compute_frag_scores_udf, schema=schema)\
            .toPandas()
    
    if save:
        file_name = os.path.basename(parquet_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        save_path = result_path + '/' + file_name_without_ext + '.tsv.gz'
        scores_df.to_csv(save_path, sep='\t', index=False)


def score_matrix_n_times(mix_dir_path, result_path, pat_cols, region_df, batch_size, schema, spark, compute_frag_scores_udf, save=False, verbose=False):
    '''
    mixture directory of replicate mixture parquets --> score matrix per replicate mixture parquet
    '''
    
    # create result directory   
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    # given directory path grab all parquet --> load path strings into a list
    list_parquet_paths = get_file_paths(mix_dir_path)
    
    # for each parquet in the list run score_matrix
    for path in list_parquet_paths:
        file_name = os.path.basename(path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        print(f'--------> Computing score matrix for {file_name_without_ext}')
            
        score_matrix(parquet_path=path,
                    result_path=result_path,
                    pat_cols=pat_cols, 
                    region_df=region_df, 
                    batch_size=batch_size, 
                    schema=schema, 
                    spark=spark,
                    compute_frag_scores_udf=compute_frag_scores_udf,
                    save=save, 
                    verbose=verbose) 
    print('\n')


def score_matrix_from_mixture_directory(path_to_mixture_dir, result_path, pat_cols, region_df, batch_size, schema, spark, compute_frag_scores_udf, save=False, verbose=False):
    '''
    dir_path_to_experiment
    '''
    print(f'>>> Start computing score matrices <<< \n')
    
    # create result directory
    result_dir_path = result_path + 'methyl_score/'
    
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)
        
    # given directory path grab all mixture directories containing parquet
    list_mixture_dir_paths = get_file_paths(path_to_mixture_dir)
    
    # iterate through each mixture proportion directory
    for path in list_mixture_dir_paths:
        
        mixture_dir_name = os.path.basename(path)
        file_name_without_ext = os.path.splitext(mixture_dir_name)[0]
        save_path = result_dir_path + file_name_without_ext + '/'
        
        print(f'--> {file_name_without_ext}')

        score_matrix_n_times(mix_dir_path=path, 
                             result_path=save_path,
                             pat_cols=pat_cols, 
                             region_df=region_df, 
                             batch_size=batch_size, 
                             schema=schema, 
                             spark=spark,
                             compute_frag_scores_udf=compute_frag_scores_udf,
                             save=save, 
                             verbose=verbose)
        
    print('>>> Complete. <<< \n')
    
    
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