#!/usr/bin/env python
# coding: utf-8

# In[Load Packages]:
# Basic code for Artificial Neural Network for tensorflow lite
# September 2021


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
#import the dataset
raw_dataset1 = pd.read_csv('C:/Users/<path>/Documents/ML_CDD_simeval/CDD/Merged_datasets_cdd.csv')

true_labels = raw_dataset1.pop('dofv')

rawdat1 = raw_dataset1.copy()
is_test1 = rawdat1['Model_number']=='DMN343'
rawdat1 = rawdat1[is_test1]
rawdat1.drop(['ID', 'Study_ID', 'Model_number', 'lin_model'], 
                 axis = 1, 
                 inplace = True) 

# change to float32 for tflite
rawdat2 = rawdat1.copy()
rawdat2 = rawdat2.astype(np.float32)


# In[Load tflite model and predict outliers (residuals)]:

input_data = rawdat2 
    
def predict_outliers(model): 
    interpreter = tf.lite.Interpreter(model_path='C:/Users/<path>/Documents/ML_CDD_simeval/ml-devel/infl_indiv/SC_testing/infl_test.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    nrows = len(input_data)
    output = np.empty(nrows)
    
    for i in range(0,nrows):
        interpreter.set_tensor(input_details[0]['index'], input_data[i : (i +1), :])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output[i]= output_data
    
    return(output, input_details, output_details)

print(output)

