#!/usr/bin/env python
# coding: utf-8

# In[Load Packages]:
# Basic code for Artificial Neural Network for tensorflow lite
# September 2021
# adapted from Alzahra Hamdan (August 2021) 

# load tensorflow and keras
#import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import models, layers, optimizers, datasets
# from tensorflow.keras.layers.experimental import preprocessing

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, roc_curve, auc
# from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import precision_recall_curve, f1_score


#helper libraries
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make numpy printouts easier to read.
#np.set_printoptions(precision=3, suppress=True)

#print(tf.__version__)


import tflite_runtime.interpreter as tflite
#interpreter = tflite.Interpreter(model_path=model_path)


# In[Generate test data]:
raw_data = pd.read_csv('merged_datasets_for_simeval.csv')


rawdat1 = raw_data.copy()
is_test1 = rawdat1['Model_number']=='PRZrun4'
rawdat1 = rawdat1[is_test1]
rawdat1.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model', 'residual'], 
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


# In[Load tflite model and predict outliers (residuals)]:
input_data = rawdat2 
#output = np.array([])
#nrows = len(rawdat2)
output = np.empty(len(input_data))
output = output.astype(np.float32)

model_path='ml-devel/outlier/outliers.tflite'
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
    
# for i in range(0,nrows):
#     interpreter.set_tensor(input_details[0]['index'], input_data[i : (i +1), :])
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     output[i]= output_data


def predict_outliers(input_data, model_path): 
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details( )
    output_details = interpreter.get_output_details( )
    
    #nrows = len(input_data)  
    # print('----------------------------------------------------------')
    # print(f'Input details: {input_details}')
    # print('----------------------------------------------------------')
    # print(f'Output details: {output_details}')
    # print('----------------------------------------------------------')
    #output = np.empty(len(input_data))
    #output = output.astype(np.float32)
        
    for i in range(0,len(input_data)):
        interpreter.set_tensor(input_details[1]['index'], input_data[i : (i +1), :])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[1]['index'])
        output = output_data()
        
    return(output)
    

out = predict_outliers(rawdat2, model_path)


