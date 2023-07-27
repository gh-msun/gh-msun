import numpy as np
import random


def one_to_many_seeds(seed, n):
    '''
    Generate seeds (between 0 and 1 million)
    '''
    random.seed(seed)
    seeds = [random.randint(0, 10**6) for _ in range(n)]
    return(seeds)


def titration_reordering(list_of_cell_types, titrating_cell_type):
    '''
    Function to reorder the list of cell types such that the titrating cell
    type comes first in order in the list.
    '''
    if titrating_cell_type in list_of_cell_types:
        list_of_cell_types.remove(titrating_cell_type)
    list_of_cell_types.insert(0, titrating_cell_type)
    return list_of_cell_types


def generate_uniform_background_proportions(titration_list, cell_types):
    '''
    '''
    def punif(p, n):
        return((1-p)/n)
    
    n_remain = len(cell_types)-1 
    proportions = []

    for i, p in enumerate(titration_list):
        proportion_list = list(np.repeat(punif(p, n_remain), n_remain))
        proportion_list.insert(0, p)
        proportions.append(proportion_list)
        
    return(proportions)


def generate_custom_background_proportion(p, custom_proportion, list_of_celltypes, titrating_celltype):

    # create list of background celltypes
    other_celltypes = list_of_celltypes.copy()  
    other_celltypes.remove(titrating_celltype)
    
    # normalizing value s.t. proportions sum to 1 for (1-p)
    custom_proportion_ = {k: v for k, v in custom_proportion.items() if k != titrating_celltype}
    total = sum(custom_proportion_.values())

    relative_proportion = []
    
    # normalize so background sum to 1-p
    for key in other_celltypes:
        pnew = (custom_proportion[key] / total) * (1-p)
        relative_proportion.append(pnew)

    relative_proportion.insert(0, p)
    
    return(relative_proportion)


def generate_custom_background_proportions(titration_list, custom_proportion, list_of_celltypes, titrating_celltype):
    
    proportions = []
    
    for p in titration_list:
        proportion = generate_custom_background_proportion(p=p, 
                                      custom_proportion=custom_proportion, 
                                      list_of_celltypes=list_of_celltypes, 
                                      titrating_celltype=titrating_celltype)
        proportions.append(proportion)
        
    return(proportions)


def reorder_abridged_name(unordered_celltypes, ordered_celltypes, unordered_abridged_names):
    
    list1 = unordered_celltypes
    list2 = unordered_abridged_names
    list1_ordered = ordered_celltypes
    
    # Zip the original list1 and list2 together
    zipped_pairs = zip(list1, list2)

    # Sort the zipped pair based on list1_rearranged
    pairs_sorted_by_list1 = sorted(zipped_pairs, key=lambda x: list1_ordered.index(x[0]))

    # Unzip the pairs to get list2 rearranged
    list1_sorted, list2_sorted = zip(*pairs_sorted_by_list1)

    return(list(list2_sorted)) 