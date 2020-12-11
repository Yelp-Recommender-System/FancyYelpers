#!/usr/bin/env python
# coding: utf-8

# # Long Format to Sparse

import numpy as np
import pandas as pd


def long_format_to_sparse(data, pre_feature):

    user_id = np.load('../data/user_id_lst.npy')
    user_id_lst = user_id.tolist()
    busi_id = np.load('../data/busi_id_lst.npy')
    busi_id_lst = busi_id.tolist()
    
    test_sparse_matrix = np.zeros(shape=(len(user_id_lst), len(busi_id_lst)))
    for i in range(len(data)):
        predict_col_index = data.columns.get_loc(pre_feature)
        predict_ratings = data.iloc[i, predict_col_index]
        row_index = user_id_lst.index(data.iloc[i, 0]) # user_id
        column_index = busi_id_lst.index(data.iloc[i, 1]) # business_id
        
        test_sparse_matrix[row_index, column_index] = predict_ratings
        
    return test_sparse_matrix


# # Sparse to Long Format
def sparse_to_long_format(sparse_matrix):
    
    user_id_lst = np.load('../data/user_id_lst.npy')
    busi_id_lst = np.load('../data/busi_id_lst.npy')
    
    user_loc_lst = np.nonzero(sparse_matrix)[0]
    busi_loc_lst = np.nonzero(sparse_matrix)[1]
    
    prediction = [nlp_sparse[loc] for loc in zip(user_loc_lst, busi_loc_lst)]
    
    user_id = [user_id_lst[i] for i in user_loc_lst]
    busi_id = [busi_id_lst[i] for i in busi_loc_lst]
    
    long_format = pd.DataFrame({'user_id': user_id,
                               'busi_id': busi_id,
                               'prediction_ratings': prediction})
    
    return long_format




