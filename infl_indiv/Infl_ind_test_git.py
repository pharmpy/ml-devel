#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 03:02:12 2021

@author: Osama Qutishat
Modified 09th September 2021 
@author: Simon Carter
"""


# In[1]:


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


# In[3]:


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


# In[4]:


#import the dataset
raw_dataset = pd.read_csv('C:/Users/simca176/Documents/ML_CDD_simeval/CDD/Merged_datasets_cdd.csv')
print(raw_dataset.describe())
X_all = raw_dataset.copy()

# log the OFV ratio as range goes from 2e-3 to 2e7
X_all['logOFVratio'] = np.log(X_all.pop('OFVRatio'))
X_all.drop(['ID','Study_ID', 'Model_number'],  axis=1, inplace=True)
#X_all = X_all[X_all['lin_model']== 0]
x_all = X_all.copy()
# labels for prediction
Y_all = X_all.pop('dofv')
#x_all = X_all
# normalisation
#scaler = StandardScaler().fit(X_all)
#x_all = scaler.transform(X_all)

print(X_all.head)
# In[5]:


# model for cdd prediction
# inputs = all predictors, targets = what training
#inputs1 = x_all
#targets = Y_all


# stratified K fold cross validation - for linear see bottom of file

# ## streamline code

# In[6]:


lr = 0.00007

num_folds = 10     
sig_value = 3.84  # significance value for influential individual

# Stratify data to keep ratios same across each split
StratifiedkFold = StratifiedKFoldReg(n_splits=num_folds, shuffle=True, random_state = random_seed)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=x_all[0].shape))
model.add(layer = tf.keras.layers.Normalization(axis=1))
model.add(layer.adapt(x_all))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(36, activation='relu'))
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(12, activation='relu'))
model.add(tf.keras.layers.Dense(1))
#layer.adapt(x_all)

# def create_model():
#     model = tf.keras.models.Sequential([
#     keras.layers.Input(input_shape=x_all[0].shape), 
#     layer.adapt(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(36, activation='relu'),
#     keras.layers.Dense(24, activation='relu'),
#     keras.layers.Dense(12, activation='relu'),
#     keras.layers.Dense(1)
#   ])

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00007), loss='mse')
    # return model

# Create a basic model instance
#model = create_model()

# Display the model's architecture
model.summary()


# In[8]:


# Set up empty vectors for outputs
loss_per_fold = []           # to store test loss value in each fold
Train_loss_per_fold = []     # to store training loss value in each fold
predicted_y = np.array([])   # to store predicted residual value from each CV fold
true_y = np.array([]) 

true_pos = []
true_neg = []
false_pos = []
false_neg = []
ave_precision = []

fold_no = 1
for train, test in StratifiedkFold.split(X_all, Y_all):
          
    # Fit data to model
    history = model.fit(X_all[train], Y_all[train], validation_data=(X_all[test], Y_all[test]), epochs=200, verbose=0)
    #plot_history(history, 'ANN1')
            
    #to store values for plotting global predicted vs. true residual values
    test_predictions = model.predict(X_all[test]).flatten()
    predicted_y = np.append(predicted_y, test_predictions)
        
    y_test_array = Y_all[test].values
    true_y = np.append(true_y, y_test_array)  
         
    # Print fold_no during training
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    #print('Average precision-recall score (binary): {0:0.2f}'.format(average_precision))
    print('------------------------------------------------------------------------')
    
    # Generate generalization metrics
    scores = model.evaluate(inputs1[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names} of {scores}')
    scores_training = model.evaluate(inputs1[train], targets[train], verbose=0)
    print(f'Training Score for fold {fold_no}: {model.metrics_names} of {scores_training}')
    loss_per_fold.append(scores)
    Train_loss_per_fold.append(scores_training)

    # Increase fold number
    fold_no = fold_no + 1


# In[10]:


print('----------------------- Influential Individual Summary ---------------------------------------')
print('4 hidden layers (Nodes: 64, 32, 24, 12)')
print(f'learning rate: {lr}, number of epochs: 200')
print(f'Average scores for all {fold_no} folds:')
print(f'Testing Loss: {np.mean(loss_per_fold): 0.4f} ({min(loss_per_fold):0.4f}-{max(loss_per_fold):0.4f})')
print(f'Training Loss: {np.mean(Train_loss_per_fold): 0.4f} ({min(Train_loss_per_fold):0.4f}-{max(Train_loss_per_fold):0.4f})')
print('--------------------------------------------------------------')


# In[12]:


# save model
model.save('<path>/Documents/CDD/influ_indiv_mods')


# In[13]:


# test on dataset
new_model = tf.keras.models.load_model('<path>/Documents/CDD/influ_indiv_mods')
new_model.summary()


# In[29]:


#import the dataset
raw_dataset1 = pd.read_csv('Merged_datasets_cdd.csv')

# log the OFV ratio as range goes from 2e-3 to 2e7
true_labels = raw_dataset1.pop('dofv')

raw_dataset1.drop(['ID', 'Study_ID', 'Model_number'],  axis=1, inplace=True)
#X_all = X_all[X_all['lin_model']== 0]

raw_test1 = scaler.transform(raw_dataset1)
raw_test1.shape
print(f'{max(true_labels)}')


# In[32]:


test1 = pd.DataFrame(new_model.predict(raw_test1))

test1.columns = ['pred_dofv']


# In[33]:


# global plot true vs. predicted
a = plt.axes(aspect='equal')
plt.scatter(x=test1, y=true_labels)
plt.xlabel('Predictions [dofv]')
plt.ylabel('True Values [dofv]')
lims = [0, 15]
plt.xlim(lims)
plt.ylim(lims)
prediction_plot = plt.show(lims, lims)


# ## Convert model to tflite for testing

# In[34]:


# save current model as saved_model:

tf.saved_model.save(model,'<path>/Documents/CDD/influ_indiv_totflite')


# In[35]:


tflite_model = tf.lite.TFLiteConverter.from_saved_model('<path>/Documents/CDD/influ_indiv_totflite').convert()
with open('<path>/Documents/CDD/influ_indiv_totflite/model.tflite', 'wb') as f:
    f.write(tflite_model)


# In[ ]:




