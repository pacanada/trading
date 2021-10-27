import numpy as np
def get_split_indexes(split_size, n_splits, total_size):
    """For splitting the dataset in chunks"""
    first_indexes = np.random.randint(total_size-split_size,size=n_splits)
    index_list = [list(range(init_index, init_index+split_size)) for init_index in first_indexes]
    # one dimension
    list_indexes = [val for lst in index_list for val in lst]
    return list_indexes

def add_training_type(df, split_size, n_splits):
    """Adding type of training data to a dataframe"""
    
    list_indexes = get_split_indexes(split_size, n_splits, df.shape[0])
    df["type"] = "training"
    df.loc[np.array(list_indexes), "type"] = "validation"
    # Just in case
    df.loc[df.index>(df.shape[0]-split_size),"type"]="validation_unseen"
    return df