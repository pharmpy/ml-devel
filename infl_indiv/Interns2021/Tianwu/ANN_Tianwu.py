#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import used Module 
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, datasets
from tensorflow.keras.layers.experimental import preprocessing


# Helper libraries
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.inspection import permutation_importance

from scipy.stats import ks_2samp

print(tf.__version__)


# In[2]:


# define used function (function for plot)
def plot_history(model_history, model_name):
    fig = plt.figure(figsize=(15,5), facecolor='w')
    ax = fig.add_subplot(121)
    ax.plot(model_history.history['loss'])
    ax.plot(model_history.history['val_loss'])
    ax.set(title=model_name + ': Model loss', ylabel='Loss', xlabel='Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')
    ax = fig.add_subplot(122)
    ax.plot(np.log(model_history.history['loss']))
    ax.plot(np.log(model_history.history['val_loss']))
    ax.set(title=model_name + ': Log model loss', ylabel='Log loss', xlabel='Epoch')
    ax.legend(['Train', 'Test'], loc='upper right')
    plt.show()
    plt.close()    


# In[3]:


# import the data 
data_all = pd.read_csv("Merged_dataset_cdd.csv") # change the wd and data name
data_all


# In[4]:


# separate out X and Y
dofv_all_Y = data_all[['dofv']]  # this is the target (Y)

# this is the X data. When change the data file, should check if this Slices is right or not!!
data_all_X = data_all.loc[:, "Model_subjects":"lin_model"] 
data_all_X


# In[5]:


#1
############################################### Try Classic KFold
# Define the K-fold Cross Validator
kfold = KFold(n_splits= 5, shuffle=True, random_state= 1234)

#set hyperparmaeter

lr = 0.000075
Batch_size = 100
Epochs = 200


# In[6]:


# K-fold Cross Validation model evaluation
fold_no = 1
# Define per-fold score containers
loss_per_fold = []

for train, test in kfold.split(data_all_X):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
  
    X_train_0 = tf.constant(data_all_X.iloc[train])
    X_test_0 = tf.constant(data_all_X.iloc[test])
    Y_train = tf.constant(dofv_all_Y.iloc[train])
    Y_test = tf.constant(dofv_all_Y.iloc[test])
    
    # standardize X_train0 and X_test0 to give X_train and X_test
    scaler = StandardScaler().fit(X_train_0)
    X_train = scaler.transform(X_train_0)
    X_test = scaler.transform(X_test_0)
    
    # Define the model architecture
    inps = layers.Input(shape=X_train[0].shape)
    x = layers.Dense(12, activation='relu')(inps)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(1)(x)
    ANN2 = models.Model(inputs=inps, outputs=preds)

    ANN2.compile(optimizer=optimizers.RMSprop(learning_rate=lr), loss='mse')
    history = ANN2.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                      batch_size = Batch_size,
                     epochs=Epochs, verbose=0
                     ,shuffle = True
                    ) 
    plot_history(history, 'ANN2')

    # Generate generalization metrics
    MSE_scores = ANN2.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for fold {fold_no}: {ANN2.metrics_names[0]} of {MSE_scores}')
    loss_per_fold.append(MSE_scores)

    # Increase fold number
    fold_no = fold_no + 1


# In[7]:


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[8]:


# check the feature importance 

# Use the 5th folder train and test to find the feature importance 
r = permutation_importance(ANN2, X_train, Y_train,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{data_all_X.columns.tolist()[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")


# In[10]:


#2
###################### try the stratifiedFold
# make a binary Y_value for StratifiedKFoldReg (>3 is 1 while <3 is 0)
binary_odfv = []
for i in range(len(dofv_all_Y)):
    if dofv_all_Y.iloc[i,0] > 3:
        binary_odfv.append(1)
    else:
        binary_odfv.append(0)

skf = StratifiedKFold(n_splits=5,random_state=2020, shuffle=True)


# In[11]:


plt.hist(dofv_all_Y)


# In[12]:


# K-fold Cross Validation model evaluation
fold_no = 1
# Define per-fold score containers
loss_per_fold = []

for train, test in skf.split(data_all_X, binary_odfv):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
  
    X_train_0 = tf.constant(data_all_X.iloc[train])
    X_test_0 = tf.constant(data_all_X.iloc[test])
    Y_train = tf.constant(dofv_all_Y.iloc[train])
    Y_test = tf.constant(dofv_all_Y.iloc[test])
    
    # standardize X_train0 and X_test0 to give X_train and X_test
    scaler = StandardScaler().fit(X_train_0)
    X_train = scaler.transform(X_train_0)
    X_test = scaler.transform(X_test_0)
    
    # Define the model architecture
    inps = layers.Input(shape=X_train[0].shape)
    x = layers.Dense(12, activation='relu')(inps)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(1)(x)
    ANN2 = models.Model(inputs=inps, outputs=preds)

    ANN2.compile(optimizer=optimizers.RMSprop(learning_rate=lr), loss='mse')
    history = ANN2.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                        batch_size = Batch_size, 
                       epochs=Epochs, verbose=0, shuffle = True) 
    plot_history(history, 'ANN2')

    # Generate generalization metrics
    MSE_scores = ANN2.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for fold {fold_no}: {ANN2.metrics_names[0]} of {MSE_scores}')
    loss_per_fold.append(MSE_scores)

    # Increase fold number
    fold_no = fold_no + 1


# In[13]:


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[14]:


# check the feature importance 
# Use the 5th folder train and test to find the feature importance 
r = permutation_importance(ANN2, X_train, Y_train,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{data_all_X.columns.tolist()[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")


# In[16]:


#3
################# combine with using decreasing learning rate
# Piecewise Constant Decay
boundaries=[350, 700, 1400, 2100]  #  0-25 25-50 50-100 100-150 150-inf 
values=[0.0005, 0.0003, 0.0001, 0.00009, 0.000075]  # the learning rate in each region
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                            boundaries=boundaries, values=values, name=None)

# K-fold Cross Validation model evaluation
fold_no = 1
# Define per-fold score containers
loss_per_fold = []

for train, test in skf.split(data_all_X, binary_odfv):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
  
    X_train_0 = tf.constant(data_all_X.iloc[train])
    X_test_0 = tf.constant(data_all_X.iloc[test])
    Y_train = tf.constant(dofv_all_Y.iloc[train])
    Y_test = tf.constant(dofv_all_Y.iloc[test])
    
    # standardize X_train0 and X_test0 to give X_train and X_test
    scaler = StandardScaler().fit(X_train_0)
    X_train = scaler.transform(X_train_0)
    X_test = scaler.transform(X_test_0)
    
    # Define the model architecture
    inps = layers.Input(shape=X_train[0].shape)
    x = layers.Dense(12, activation='relu')(inps)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(1)(x)
    ANN2 = models.Model(inputs=inps, outputs=preds)

    ANN2.compile(optimizer=optimizers.RMSprop(learning_rate=lr), loss='mse')
    history = ANN2.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                        batch_size = Batch_size, 
                       epochs=Epochs, verbose=0, shuffle = True) 
    plot_history(history, 'ANN2')

    # Generate generalization metrics
    MSE_scores = ANN2.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for fold {fold_no}: {ANN2.metrics_names[0]} of {MSE_scores}')
    loss_per_fold.append(MSE_scores)

    # Increase fold number
    fold_no = fold_no + 1


# In[17]:


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[18]:


# check the feature importance 
# Use the 5th folder train and test to find the feature importance 
r = permutation_importance(ANN2, X_train, Y_train,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{data_all_X.columns.tolist()[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")


# In[20]:


#4
################ optimized using Adam
# K-fold Cross Validation model evaluation
fold_no = 1
# Define per-fold score containers
loss_per_fold = []

for train, test in skf.split(data_all_X, binary_odfv):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
  
    X_train_0 = tf.constant(data_all_X.iloc[train])
    X_test_0 = tf.constant(data_all_X.iloc[test])
    Y_train = tf.constant(dofv_all_Y.iloc[train])
    Y_test = tf.constant(dofv_all_Y.iloc[test])
    
    # standardize X_train0 and X_test0 to give X_train and X_test
    scaler = StandardScaler().fit(X_train_0)
    X_train = scaler.transform(X_train_0)
    X_test = scaler.transform(X_test_0)
    
    # Define the model architecture
    inps = layers.Input(shape=X_train[0].shape)
    x = layers.Dense(12, activation='relu')(inps)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(1)(x)
    ANN2 = models.Model(inputs=inps, outputs=preds)

    ANN2.compile(optimizer=optimizers.Adam(learning_rate=0.00008), loss='mse')
    history = ANN2.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                        batch_size = Batch_size, 
                       epochs=Epochs, verbose=0, shuffle = True) 
    plot_history(history, 'ANN2')

    # Generate generalization metrics
    MSE_scores = ANN2.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for fold {fold_no}: {ANN2.metrics_names[0]} of {MSE_scores}')
    loss_per_fold.append(MSE_scores)

    # Increase fold number
    fold_no = fold_no + 1


# In[21]:


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[39]:


#5
# This is the final model, can only re-run this model (several times by different random seed and to get average)
################ optimized using Adam
# Also Piecewise Constant Decay
boundaries=[350, 700, 1400, 2100]  #  epoch: 0-25 25-50 50-100 100-150 150-inf 
values=[0.001, 0.0005, 0.0001, 0.00007, 0.00005]  # the learning rate in each region
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                            boundaries=boundaries, values=values, name=None)

skf = StratifiedKFold(n_splits=5,random_state=201, shuffle=True)


# In[40]:


# Cross Validation model evaluation
fold_no = 1
# Define per-fold score containers
loss_per_fold = []           #to store test loss value in each fold
Train_loss_per_fold = []     #to store training loss value in each fold
predcited_y = np.array([])   #to store predicted residual value from each CV fold
true_y = np.array([])        #to store true residual value from each CV fold

for train, test in skf.split(data_all_X, binary_odfv):
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
  
    X_train_0 = tf.constant(data_all_X.iloc[train])
    X_test_0 = tf.constant(data_all_X.iloc[test])
    Y_train = tf.constant(dofv_all_Y.iloc[train])
    Y_test = tf.constant(dofv_all_Y.iloc[test])
    
    # standardize X_train0 and X_test0 to give X_train and X_test
    scaler = StandardScaler().fit(X_train_0)
    X_train = scaler.transform(X_train_0)
    X_test = scaler.transform(X_test_0)
    
    # Define the model architecture
    inps = layers.Input(shape=X_train[0].shape)
    x = layers.Dense(12, activation='relu')(inps)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(1)(x)
    ANN2 = models.Model(inputs=inps, outputs=preds)

    ANN2.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse')
    history = ANN2.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                        batch_size = Batch_size, 
                       epochs=Epochs, verbose=0, shuffle = True) 
    plot_history(history, 'ANN2')

    
    #to store values for plotting global predicted vs. true residual values
    test_predictions = ANN2.predict(X_test).flatten()
    predcited_y = np.append(predcited_y, test_predictions)

    true_y = np.append(true_y, Y_test)
    
    # Generate generalization metrics
    MSE_scores = ANN2.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for fold {fold_no}: {ANN2.metrics_names[0]} of {MSE_scores}')
    loss_per_fold.append(MSE_scores)
    
    scores_training = ANN2.evaluate(X_train, Y_train, verbose=0)
    print(f'Training Score for fold {fold_no}: {ANN2.metrics_names[0]} of {scores_training}')
    Train_loss_per_fold.append(scores_training)

    # Increase fold number
    fold_no = fold_no + 1


# In[41]:


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Test Loss: {loss_per_fold[i]}')
  print(f'> Fold {i+1} - Train Loss: {Train_loss_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Test Loss: {np.mean(loss_per_fold)}')
print(f'> Train Loss: {np.mean(Train_loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[42]:


# check the feature importance 
# Use the 5th folder train and test to find the feature importance 
r = permutation_importance(ANN2, X_train, Y_train,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{data_all_X.columns.tolist()[i]:<8}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")


# In[43]:


# global plot true vs. predicted
a = plt.axes(aspect='equal')
plt.scatter(predcited_y, true_y)
plt.xlabel('Predictions [dofv]')
plt.ylabel('True Values [dofv]')
lims = [0, 10]
plt.xlim(lims)
plt.ylim(lims)
prediction_plot = plt.plot(lims, lims)


# In[ ]:




