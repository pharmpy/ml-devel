# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 03:02:12 2021

@author: oqtai
"""
# import the libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models, layers, optimizers


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from sklearn.inspection import permutation_importance


# function for ploting

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
    
# Define the cross-validator object for regression, which inherits from 
# StratifiedKFold, overwritting the split method
#code source: https://colab.research.google.com/drive/1KnXujsQDvLZOgCRg_iis036cwffwZM2_?usp=sharing#scrollTo=2q_q9w8Jpmwd
# &https://github.com/scikit-learn/scikit-learn/issues/4757

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
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits) \
            for ii in range(n_labels)])
        
        # Get number of points that would fall
        # out of the equally-sized bins
        mod = np.mod(n_samples, self.n_splits)
        
        # Find unique idxs of first unique label's ocurrence
        _, labels_idx = np.unique(y_labels_sorted, return_index=True)
        
        # sample randomly the label idxs to which assign the 
        # the mod points
        rand_label_ix = np.random.choice(labels_idx, mod, replace=False)

        # insert these at the beginning of the corresponding bin
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])
        
        # find each element of y to which label corresponds in the sorted 
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label
    
        # put labels according to the given y order then
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])

        return super().split(X, y_labels, groups)
    
# import the dataset
raw_dataset = pd.read_csv('Merged_datasets_cdd.csv')
X_all = raw_dataset.copy()
X_all.drop(['ID', 'Study_ID', 'Model_number', 'residual'],  axis=1, inplace=True)
Y_all = X_all.pop('dofv')

#normalization
scaler = StandardScaler().fit(X_all)
x_all = scaler.transform(X_all)

# model for cdd prediction
inputs1= x_all
targets = Y_all

num_folds = 10               #number of folds for crosvalidation
loss_per_fold = []           #to store test loss value in each fold
Train_loss_per_fold = []     #to store training loss value in each fold
predcited_y = np.array([])   #to store predicted residual value from each CV fold
true_y = np.array([])        #to store true residual value from each CV fold



StratifiedkFold = StratifiedKFoldReg(n_splits=num_folds, shuffle=True)
fold_no = 1
for train, test in StratifiedkFold.split(inputs1, targets):
    #Define and summarize the model:
    inps = layers.Input(shape=inputs1[train][0].shape)
    x = layers.Dense(64, activation='relu')(inps)
    x = layers.Dense(36, activation='relu')(x)
    x = layers.Dense(24, activation='relu')(x)
    x = layers.Dense(12, activation='relu')(x)

    preds = layers.Dense(1)(x)

    ANN1 = models.Model(inputs=inps, outputs=preds)

    #Compile the model
    lr = 0.00007
    ANN1.compile(optimizer=optimizers.RMSprop(lr=lr), loss='mse')
    
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = ANN1.fit(inputs1[train], targets[train], validation_data=(inputs1[test], targets[test]), epochs=200, verbose=0)
    plot_history(history, 'ANN1')
    
        
    #to store values for plotting global predicted vs. true residual values
    test_predictions = ANN1.predict(inputs1[test]).flatten()
    predcited_y = np.append(predcited_y, test_predictions)

    y_test_array = targets[test].values
    true_y = np.append(true_y, y_test_array)

    
    # Generate generalization metrics
    scores = ANN1.evaluate(inputs1[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {ANN1.metrics_names} of {scores}')

    scores_training = ANN1.evaluate(inputs1[train], targets[train], verbose=0)
    print(f'Training Score for fold {fold_no}: {ANN1.metrics_names} of {scores_training}')

    loss_per_fold.append(scores)
    Train_loss_per_fold.append(scores_training)

    # Increase fold number
    fold_no = fold_no + 1

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Testing Loss: {np.mean(loss_per_fold)}')
print(f'> Training Loss: {np.mean(Train_loss_per_fold)}')
print('------------------------------------------------------------------------')

# global plot true vs. predicted

a = plt.axes(aspect='equal')
plt.scatter(predcited_y, true_y)
plt.xlabel('Predictions [dofv]')
plt.ylabel('True Values [dofv]')
lims = [0, 10]
plt.xlim(lims)
plt.ylim(lims)
prediction_plot = plt.plot(lims, lims)


# permutation importance
r = permutation_importance(ANN1, inputs1, targets,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{X_all.columns.tolist()[i]:<10}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")

