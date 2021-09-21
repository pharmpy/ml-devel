#!/usr/bin/env python
# coding: utf-8

# In[Load Packages]:
# Basic code for Artificial Neural Network for tensorflow lite
# September 2021
# adapted from Alzahra Hamdan (August 2021) 

# load tensorflow and keras
import tensorflow as tf
#from tensorflow.python.platform import gfile
#from tensorflow.python.framework import tensor_util

from tensorflow import keras
# from tensorflow.keras import models, layers, optimizers, datasets
# from tensorflow.keras.layers.experimental import preprocessing

#helper libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json
import h5py

# Make numpy printouts easier to read.
#np.set_printoptions(precision=3, suppress=True)

#print(tf.__version__)

#%%

# # In[Read in .pb model and convert to csvs per layer (weights and bias)]:
# # Code taken from: https://stackoverflow.com/questions/56260192/load-tensorflow-model-without-importing-tensorflow

# #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

# #config = tf.ConfigProto(allow_soft_placement=True,
# #                        log_device_placement=True)

# GRAPH_PB_PATH = 'c:/Users/simca176/Documents/ML_CDD_simeval/ml-devel/outlier/saved_model.pb'
# with tf.Session(config=config) as sess:
#     print("load graph")
#     with gfile.FastGFile(GRAPH_PB_PATH, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         sess.graph.as_default()
#         tf.import_graph_def(graph_def, name='')
#         graph_nodes = [n for n in graph_def.node]
#         wts = [n for n in graph_nodes if n.op == 'Const']

# result = []
# result_name = []
# for n in wts:
#     result_name.append(n.name)
#     result.append(tensor_util.MakeNdarray(n.attr['value'].tensor))

# np.savetxt("layer1_weight.csv", result[0], delimiter=",")
# np.savetxt("layer1_bias.csv", result[1], delimiter=",")
# np.savetxt("layer2_weight.csv", result[2], delimiter=",")
# np.savetxt("layer2_bias.csv", result[3], delimiter=",")
# np.savetxt("layer1_weight.csv", result[4], delimiter=",")
# np.savetxt("layer1_bias.csv", result[5], delimiter=",")


# In[Read in model json file and weights file]:

f = open('c:/Users/simca176/Documents/ML_CDD_simeval/ml-devel/outlier_mod.json',)    
mod = json.load(f)  
print(mod)


#%%
mod_wt = h5py.File('c:/Users/simca176/Documents/ML_CDD_simeval/ml-devel/outlier_mod_wt.h5') 

weights = {}
keys = []

# study weights structure

mod_wt.visit(keys.append) # append all keys to list
for key in keys:
    if ':' in key: # contains data if ':' in key
        #print(f[key].name)
        weights[mod_wt[key].name] = mod_wt[key].value
    
    



# # taken from: https://gist.github.com/Attila94/fb917e03b04035f3737cc8860d9e9f9b 
# def read_hdf5(path):

#     weights = {}

#     keys = []
#     with h5py.File(path, 'r') as f: # open file
#         f.visit(keys.append) # append all keys to list
#         for key in keys:
#             if ':' in key: # contains data if ':' in key
#                 print(f[key].name)
#                 weights[f[key].name] = f[key].value
#     return weights

# test = read_hdf5(path)
# In[Generate test data]:
raw_data = pd.read_csv('c:/Users/simca176/.spyder-py3/merged_datasets_for_simeval.csv')


rawdat1 = raw_data.copy()
is_test1 = rawdat1['Model_number']=='PRZrun4'
rawdat1 = rawdat1[is_test1]

true_out = rawdat1.pop('residual')

rawdat1.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model'], 
                 axis = 1, 
                 inplace = True)    
# change to float32 for tflite
rawdat2 = rawdat1.copy()
rawdat2 = rawdat2.astype(np.float32)
# remove header column
#rawdat2.columns = range(rawdat2.shape[1])


# is_test2 = raw_data['Model_number']=='DMN343'
# test_dat2 = raw_data[is_test2]
# test_dat2.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model', 'residual'], 
#                axis = 1, 
#                inplace = True)

# test_dat2_32 = test_dat2.astype(np.float32)
# remove header column
#test_dat2_32.columns = range(test_dat2_32.shape[1])

#print(test_dat2.head())

# In[]:


