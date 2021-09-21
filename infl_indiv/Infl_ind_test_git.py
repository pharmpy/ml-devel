#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 03:02:12 2021

@author: Osama Qutishat
Modified 21st September 2021 
@author: Simon Carter
"""


# In[Load Packages]:


# import the libraries<br>
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models, layers, optimizers, initializers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import precision_score, average_precision_score, precision_recall_curve, f1_score
# from sklearn.metrics import mean_squared_error, roc_curve, plot_roc_curve, auc

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print(tf.__version__)


# In[Stratified KFold function]:


# Define the cross-validator object for regression, which inherits from 
# StratifiedKFold, overwritting the split method
#code source: https://colab.research.google.com/drive/1KnXujsQDvLZOgCRg_iis036cwffwZM2_?usp=sharing#scrollTo=2q_q9w8Jpmwd
# and https://github.com/scikit-learn/scikit-learn/issues/4757

random_seed=4242
random.set_seed = random_seed
class StratifiedKFoldReg(StratifiedKFold):
    """  
    
    This class generate cross-validation partitions
    for regression setups, such that these partitions
    resemble the original sample distribution of the 
    target variable.
    
    """
    
    def split(self, X, y, groups=None):
        
        n_samples = len(y)
        
        # Number of labels to discretize our target variable,
        # into bins of quasi equal size
        n_labels = int(np.round(n_samples/self.n_splits))
        
        # Assign a label to each bin of n_splits points
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits)             for ii in range(n_labels)])
        
        # Get number of points that would fall
        # out of the equally-sized bins
        mod = np.mod(n_samples, self.n_splits)
        
        # Find unique idxs of first unique label's ocurrence
        _, labels_idx = np.unique(y_labels_sorted, return_index=True)
        
        # sample randomly the label idxs to which assign the 
        # the mod points
        rand_label_ix = np.random.choice(labels_idx, mod, replace=False)

        # insert these at the beginning of the corresponding bin
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, 
                                    y_labels_sorted[rand_label_ix])
        
        # find each element of y to which label corresponds in the sorted 
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label
    
        # put labels according to the given y order then
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])
        return super().split(X, y_labels, groups)


# In[Import dataset]:


#import the dataset
raw_dataset = pd.read_csv('Merged_datasets_cdd.csv')
print(raw_dataset.describe())
X_all = raw_dataset.copy()
#X_all.reset_index(drop=True)
# log the OFV ratio as range goes from 2e-3 to 2e7
#X_all['logOFVratio'] = np.log(X_all.pop('OFVRatio'))
X_all.drop(['ID','Study_ID', 'Model_number','lin_model'],  axis=1, inplace=True)
#X_all = X_all[X_all['lin_model']== 0]

# labels for prediction
Y = X_all.pop('dofv')
X = X_all.values

print(X_all.head)



# In[Create 4 layer ANN model]:

layer = tf.keras.layers.Normalization(axis=1)
layer.adapt(X)

def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=X.shape), 
    layer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(36, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00007), loss='mse')
    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# In[Run Kfold Cross validation]:


# Set up empty vectors for outputs
loss_per_fold = []           # to store test loss value in each fold
Train_loss_per_fold = []     # to store training loss value in each fold
predicted_y = np.array([])   # to store predicted residual value from each CV fold
true_y = np.array([]) 

# true_pos = []
# true_neg = []
# false_pos = []
# false_neg = []
# ave_precision = []

num_folds = 10     
sig_value = 3.84  # significance value for influential individual

# Stratify data to keep ratios same across each split
cv_stratified = StratifiedKFoldReg(n_splits=num_folds, shuffle=True, random_state = random_seed)

fold_no = 1
#for train, test in StratifiedkFold.split(x_all, Y_all):
for ii, (train_index, test_index) in enumerate(cv_stratified.split(X, Y)):
    Y_train, Y_test = Y[train_index], Y[test_index]
    X_train, X_test = X[train_index], X[test_index]          
  
    # Fit data to model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, verbose=0)
            
    #to store values for plotting global predicted vs. true residual values
    test_predictions = model.predict(X_test).flatten()
    predicted_y = np.append(predicted_y, test_predictions)
        
    y_test_array = Y_test.values
    true_y = np.append(true_y, y_test_array)  
         
    # Print fold_no during training
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    print('------------------------------------------------------------------------')
    
    # Generate generalization metrics
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names} of {scores}')
    scores_training = model.evaluate(X_train, Y_train, verbose=0)
    print(f'Training Score for fold {fold_no}: {model.metrics_names} of {scores_training}')
    loss_per_fold.append(scores)
    Train_loss_per_fold.append(scores_training)

    # Increase fold number
    fold_no = fold_no + 1


# In[Summarise output]:


print('----------------------- Influential Individual Summary ---------------------------------------')
print('4 hidden layers (Nodes: 64, 32, 24, 12)')
print(f'learning rate: 0.00007, number of epochs: 200')
print(f'Average scores for all {fold_no} folds:')
print(f'Testing Loss: {np.mean(loss_per_fold): 0.4f} ({min(loss_per_fold):0.4f}-{max(loss_per_fold):0.4f})')
print(f'Training Loss: {np.mean(Train_loss_per_fold): 0.4f} ({min(Train_loss_per_fold):0.4f}-{max(Train_loss_per_fold):0.4f})')
print('----------------------------------------------------------------------------------------------')

# In[Permutation Importance]:

feature_name = ['Model_subjects', 'Model_observations',
                'Obsi_Obs_Subj', 'Covariate_relations', 'Max_cov', 'Max_CWRESi', 'Median_CWRESi',
                'Max_EBEij_omegaj', 'OFVRatio', 'mean_ETC_omega']

r = permutation_importance(model, X, Y,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
      if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
          print(f"{feature_name[i]:<10}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")


# In[Save Model]:

# save model
model.save('ml-devel/infl_indiv/SC_testing')

# convert to tflite for pharmpy testing
tflite_model = tf.lite.TFLiteConverter.from_saved_model('ml-devel/infl_indiv/SC_testing').convert()
with open('ml-devel/infl_indiv/SC_testing/infl_test.tflite', 'wb') as f:
    f.write(tflite_model)

# In[Remove variables then load model for testing]:


# test on dataset
new_model = tf.keras.models.load_model('ml-devel/infl_indiv/SC_testing')
new_model.summary()


# In[Test tensorflow model]:

#import the dataset
raw_dataset1 = pd.read_csv('Merged_datasets_cdd.csv')

# log the OFV ratio as range goes from 2e-3 to 2e7
true_labels = raw_dataset1.pop('dofv')

rawdat1 = raw_dataset1.copy()
is_test1 = rawdat1['Model_number']=='DMN343'
rawdat1 = rawdat1[is_test1]
rawdat1.drop(['ID', 'Study_ID', 'Model_number', 'lin_model'], 
                 axis = 1, 
                 inplace = True) 

# test influential individual model   
test1 = new_model.predict(rawdat1)
print(test1)


