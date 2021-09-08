#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, datasets
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, f1_score

#helper libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)


# In[ ]:


#Plotting function
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


# In[ ]:


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
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])
        
        # find each element of y to which label corresponds in the sorted 
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label
    
        # put labels according to the given y order then
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])

        return super().split(X, y_labels, groups)


# In[ ]:


#load data
raw_dataset = pd.read_csv('.spyder-py3/merged_datasets_for_simeval.csv')
dataset = raw_dataset.copy()
dataset.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model'], axis = 1, inplace=True)
dataset.head()


# In[ ]:


#split features and labels
x = dataset.copy()
y = x.pop('residual')


# In[ ]:


# normalization
scaler = StandardScaler().fit(x)
X = scaler.transform(x)


# In[ ]:


##################################################################################################################################################


# In[ ]:


#ANN5: This is the final model

n_splits = 10                #number of folds
loss_per_fold = []           #to store test loss value in each fold
Train_loss_per_fold = []     #to store training loss value in each fold
predcited_y = np.array([])   #to store predicted residual value from each CV fold
true_y = np.array([])        #to store true residual value from each CV fold

cv_stratified = StratifiedKFoldReg(n_splits=n_splits, shuffle=True, random_state=10)   # Stratified CV

fold_no = 1
for ii, (train_index, test_index) in enumerate(cv_stratified.split(X, y)):
  y_train, y_test = y[train_index], y[test_index]
  X_train, X_test = X[train_index], X[test_index]

  #Define and summarize the model
  inps = layers.Input(shape=X_train[0].shape)
  x = layers.Dense(48, activation='relu')(inps)
  x = layers.Dense(24, activation='relu')(x)
  x = layers.Dense(12, activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  preds = layers.Dense(1)(x)
  ANN5 = models.Model(inputs=inps, outputs=preds)

  #Compile the model
  lr = 0.00007
  ANN5.compile(optimizer=optimizers.RMSprop(lr=lr), loss='mse')
    
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  test_labels = y_test.to_list()
  test_labels = [round(num, 2) for num in test_labels]
  print(test_labels)   #to have a look at the true residual values for test dataset

  #print histogram of y_test and y_train
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
  
  axs.hist(y_train, label="training")
  axs.hist(y_test, label="test")
  axs.legend()
  plt.tight_layout()

  # Fit data to model
  history = ANN5.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
  plot_history(history, 'ANN5')

  
  #to store values for plotting global predicted vs. true residual values
  test_predictions = ANN5.predict(X_test).flatten()
  predcited_y = np.append(predcited_y, test_predictions)

  y_test_array = y_test.values
  true_y = np.append(true_y, y_test_array)

    
  # Generate generalization metrics
  scores = ANN5.evaluate(X_test, y_test, verbose=0)
  print(f'Test Score for fold {fold_no}: {ANN5.metrics_names} of {scores}')

  scores_training = ANN5.evaluate(X_train, y_train, verbose=0)
  print(f'Training Score for fold {fold_no}: {ANN5.metrics_names} of {scores_training}')

  loss_per_fold.append(scores)
  Train_loss_per_fold.append(scores_training)

  # Increase fold number
  fold_no = fold_no + 1

# global plot true vs. predicted

a = plt.axes(aspect='equal')
plt.scatter(predcited_y, true_y)
plt.xlabel('Predictions [residual]')
plt.ylabel('True Values [residual]')
lims = [-5, 20]
plt.xlim(lims)
plt.ylim(lims)
prediction_plot = plt.plot(lims, lims)


# In[ ]:


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Training Loss: {Train_loss_per_fold[i]} - Testing Loss: {loss_per_fold[i]} -')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Test Loss: {np.mean(loss_per_fold)}')
print(f'> Training Loss: {np.mean(Train_loss_per_fold)}')
print('------------------------------------------------------------------------')


# In[ ]:


#permutation importance
feature_name = ['Model_subjects', 'Model_observations',
                'Obsi_Obs_Subj', 'Covariate_relations', 'Max_cov', 'Max_CWRESi', 'Median_CWRESi',
                'Max_EBEij_omegaj', 'OFVRatio', 'mean_ETC_omega']

r = permutation_importance(ANN5, X, y,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
         print(f"{feature_name[i]:<10}"
               f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")


# In[ ]:


#sensitivity analysis: cut-off=3 for both true and predicted residual
true_positives = 0
false_positives = 0
true_negative = 0
false_negative = 0
for i in range(0,len(true_y)):
    if abs(true_y[i]) > 3 and abs(predcited_y[i]) > 3:
        true_positives = true_positives + 1
    if abs(true_y[i]) <= 3 and abs(predcited_y[i]) > 3:
        false_positives = false_positives + 1
    if abs(true_y[i]) <= 3 and abs(predcited_y[i]) <= 3:
        true_negative = true_negative + 1
    if abs(true_y[i]) > 3 and abs(predcited_y[i]) <= 3:
        false_negative = false_negative + 1

print(f'> TP: {true_positives}')
print(f'> FP: {false_positives}')
print(f'> TN: {true_negative}')
print(f'> FN: {false_negative}')

sensitivity = true_positives/(true_positives + false_negative)
specificity = true_negative/(true_negative + false_positives)
precision = true_positives/(true_positives + false_positives)

print(f'> Sensitivity: {round(sensitivity, 3)}')
print(f'> Specificity: {round(specificity, 3)}')
print(f'> Precision: {round(precision, 3)}')


# In[ ]:


#sensitivity analysis: cut-off=3 for true and 2.5 for predicted residual
true_positives = 0
false_positives = 0
true_negative = 0
false_negative = 0
for i in range(0,len(true_y)):
    if abs(true_y[i]) > 3 and abs(predcited_y[i]) > 2.5:
        true_positives = true_positives+1
    if abs(true_y[i]) <= 3 and abs(predcited_y[i]) > 2.5:
        false_positives = false_positives + 1
    if abs(true_y[i]) <= 3 and abs(predcited_y[i]) <= 2.5:
        true_negative = true_negative+1
    if abs(true_y[i]) > 3 and abs(predcited_y[i]) <= 2.5:
        false_negative = false_negative + 1

print(f'> TP: {true_positives}')
print(f'> FP: {false_positives}')
print(f'> TN: {true_negative}')
print(f'> FN: {false_negative}')

sensitivity = true_positives/(true_positives + false_negative)
specificity = true_negative/(true_negative + false_positives)
precision = true_positives/(true_positives + false_positives)

print(f'> Sensitivity: {round(sensitivity, 3)}')
print(f'> Specificity: {round(specificity, 3)}')
print(f'> Precision: {round(precision, 3)}')


# In[ ]:


#sensitivity analysis: cut-off=3 for true and 2.0 for predicted residual
true_positives = 0
false_positives = 0
true_negative = 0
false_negative = 0
for i in range(0,len(true_y)):
    if abs(true_y[i]) > 3 and abs(predcited_y[i]) > 2:
        true_positives = true_positives+1
    if abs(true_y[i]) <= 3 and abs(predcited_y[i]) > 2:
        false_positives = false_positives + 1
    if abs(true_y[i]) <= 3 and abs(predcited_y[i]) <= 2:
        true_negative = true_negative+1
    if abs(true_y[i]) > 3 and abs(predcited_y[i]) <= 2:
        false_negative = false_negative + 1

print(f'> TP: {true_positives}')
print(f'> FP: {false_positives}')
print(f'> TN: {true_negative}')
print(f'> FN: {false_negative}')

sensitivity = true_positives/(true_positives + false_negative)
specificity = true_negative/(true_negative + false_positives)
precision = true_positives/(true_positives + false_positives)

print(f'> Sensitivity: {round(sensitivity, 3)}')
print(f'> Specificity: {round(specificity, 3)}')
print(f'> Precision: {round(precision, 3)}')


# In[ ]:


#Convert target values to binary to plot ROC & PR curves (cutoff=3)
true_outliers_binary = true_y.copy()
predicted_outliers_binary = predcited_y.copy()

for i in range(0,len(true_outliers_binary)):
    if true_outliers_binary[i] > 3:
        true_outliers_binary[i] = 1
    else:
        true_outliers_binary[i] = 0

for i in range(0,len(predicted_outliers_binary)):
    if predicted_outliers_binary[i] > 3:
        predicted_outliers_binary[i] = 1
    else:
        predicted_outliers_binary[i] = 0


# In[ ]:


# ROC curve
fpr, tpr, threshold = roc_curve(true_outliers_binary, predicted_outliers_binary)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


# precision-recall curve and f1

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

lr_precision, lr_recall, _ = precision_recall_curve(true_outliers_binary, predicted_outliers_binary)
lr_f1, lr_auc = f1_score(true_outliers_binary, predicted_outliers_binary), auc(lr_recall, lr_precision)

# summarize scores
print('f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

# plot the precision-recall curves
no_skill = len(true_outliers_binary[true_outliers_binary==1]) / len(true_outliers_binary)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='model')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# In[ ]:


#plot ROC curve at different cut-off values (0.5-3.0)
cutoff_pred = 0    # will be increased gradually in the for loop (+0.5/loop)
cutoff_true = 3    # constant
true_outliers_binary = true_y.copy()            #to convert contineous value into binary (using cutoff_true)
predicted_outliers_binary = predcited_y.copy()  #to convert contineous value into binary (using cutoff_pred)

fpr = {}            # to store fpr values from each loop with distinct variable name
tpr = {}            # same
threshold = {}      # same
roc_auc= {}         # same

for k in range(0,6):
    true_outliers_binary = true_y.copy()
    predicted_outliers_binary = predcited_y.copy()
    cutoff_pred= cutoff_pred + 0.5

    for i in range(0,len(true_outliers_binary)):
        if true_outliers_binary[i] > cutoff_true:
            true_outliers_binary[i] = 1
        else:
            true_outliers_binary[i] = 0

    for i in range(0,len(predicted_outliers_binary)):
        if predicted_outliers_binary[i] > cutoff_pred:
            predicted_outliers_binary[i] = 1
        else:
            predicted_outliers_binary[i] = 0

    print(cutoff_pred)

    fpr["fpr%s" %k], tpr["tpr%s" %k],threshold["threshold%s" %k] = roc_curve(true_outliers_binary, predicted_outliers_binary)
    roc_auc["roc_auc%s" %k] = auc(fpr["fpr%s" %k], tpr["tpr%s" %k])

plt.style.use('seaborn')
plt.title('Receiver Operating Characteristic')

plt.plot(fpr['fpr0'], tpr['tpr0'], linestyle='--',color='orange', label='cutoff=0.5'+',AUC = %0.2f' % roc_auc['roc_auc0'])
plt.plot(fpr['fpr1'], tpr['tpr1'], linestyle='--',color='green', label='cutoff=1.0'+',AUC = %0.2f' % roc_auc['roc_auc1'])
plt.plot(fpr['fpr2'], tpr['tpr2'], linestyle='--',color='blue', label='cutoff=1.5'+',AUC = %0.2f' % roc_auc['roc_auc2'])
plt.plot(fpr['fpr3'], tpr['tpr3'], linestyle='--',color='red', label='cutoff=2.0'+',AUC = %0.2f' % roc_auc['roc_auc3'])
plt.plot(fpr['fpr4'], tpr['tpr4'], linestyle='--',color='black', label='cutoff=2.5'+',AUC = %0.2f' % roc_auc['roc_auc4'])
plt.plot(fpr['fpr5'], tpr['tpr5'], linestyle='--',color='yellow', label='cutoff=3.0'+',AUC = %0.2f' % roc_auc['roc_auc5'])
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r-')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


    


# In[ ]:


# precision-recall curve and f1

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

cutoff_pred = 0
cutoff_true = 3
true_outliers_binary = true_y.copy()
predicted_outliers_binary = predcited_y.copy()

lr_precision = {}
lr_recall = {}
lr_f1 = {}
lr_auc= {}

for k in range(0,6):
    true_outliers_binary = true_y.copy()
    predicted_outliers_binary = predcited_y.copy()
    cutoff_pred= cutoff_pred + 0.5

    for i in range(0,len(true_outliers_binary)):
        if true_outliers_binary[i] > cutoff_true:
            true_outliers_binary[i] = 1
        else:
            true_outliers_binary[i] = 0

    for i in range(0,len(predicted_outliers_binary)):
        if predicted_outliers_binary[i] > cutoff_pred:
            predicted_outliers_binary[i] = 1
        else:
            predicted_outliers_binary[i] = 0

    print(cutoff_pred)

    lr_precision["lr_precision%s" %k], lr_recall["lr_recall%s" %k], _ = precision_recall_curve(true_outliers_binary, predicted_outliers_binary)
    lr_f1["lr_f1%s" %k], lr_auc["lr_auc%s" %k] = f1_score(true_outliers_binary, predicted_outliers_binary), auc(lr_recall["lr_recall%s" %k], lr_precision["lr_precision%s" %k])


plt.style.use('seaborn')
plt.title('PR plot')
plt.plot(lr_recall['lr_recall0'], lr_precision['lr_precision0'], linestyle='--',color='orange', label='cutoff=0.5'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc0'], lr_f1['lr_f10']))
plt.plot(lr_recall['lr_recall1'], lr_precision['lr_precision1'], linestyle='--',color='green', label='cutoff=1.0'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc1'], lr_f1['lr_f11']))
plt.plot(lr_recall['lr_recall2'], lr_precision['lr_precision2'], linestyle='--',color='blue', label='cutoff=1.5'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc2'], lr_f1['lr_f12']))
plt.plot(lr_recall['lr_recall3'], lr_precision['lr_precision3'], linestyle='--',color='red', label='cutoff=2.0'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc3'], lr_f1['lr_f13']))
plt.plot(lr_recall['lr_recall4'], lr_precision['lr_precision4'], linestyle='--',color='black', label='cutoff=2.5'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc4'], lr_f1['lr_f14']))
plt.plot(lr_recall['lr_recall5'], lr_precision['lr_precision5'], linestyle='--',color='yellow', label='cutoff=3.0'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc5'], lr_f1['lr_f15']))
plt.legend(loc = 'lower left')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# In[ ]:


#ANN5 with ROC curve for each fold

n_splits = 10                #number of folds
loss_per_fold = []           #to store test loss value in each fold
Train_loss_per_fold = []     #to store training loss value in each fold
predcited_y = np.array([])   #to store predicted residual value from each CV fold
true_y = np.array([])        #to store true residual value from each CV fold

tprs = []      #to store values
fprs = []
aucs = []

fpr = {}      #to store values with different variable name in each fold
tpr = {}
threshold = {}
roc_auc= {}

lr_precision = {}
lr_recall = {}
lr_f1 = {}
lr_auc= {}

cv_stratified = StratifiedKFoldReg(n_splits=n_splits, shuffle=True, random_state=10)   # Stratified CV

fold_no = 1
for ii, (train_index, test_index) in enumerate(cv_stratified.split(X, y)):
  y_train, y_test = y[train_index], y[test_index]
  X_train, X_test = X[train_index], X[test_index]

  #Define and summarize the model
  inps = layers.Input(shape=X_train[0].shape)
  x = layers.Dense(48, activation='relu')(inps)
  x = layers.Dense(24, activation='relu')(x)
  x = layers.Dense(12, activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  preds = layers.Dense(1)(x)
  ANN5 = models.Model(inputs=inps, outputs=preds)

  #Compile the model
  lr = 0.00007
  ANN5.compile(optimizer=optimizers.RMSprop(lr=lr), loss='mse')
    
     
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  test_labels = y_test.to_list()
  test_labels = [round(num, 2) for num in test_labels]
  print(test_labels)   #to have a look at the true residual values for test dataset

  #print histogram of y_test and y_train
  fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
  
  axs.hist(y_train, label="training")
  axs.hist(y_test, label="test")
  axs.legend()
  plt.tight_layout()

  # Fit data to model
  history = ANN5.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
  plot_history(history, 'ANN5')

  
  #to store values for plotting global predicted vs. true residual values
  test_predictions = ANN5.predict(X_test).flatten()
  predcited_y = np.append(predcited_y, test_predictions)

  y_test_array = y_test.values
  true_y = np.append(true_y, y_test_array)

    
  # Generate generalization metrics
  scores = ANN5.evaluate(X_test, y_test, verbose=0)
  print(f'Test Score for fold {fold_no}: {ANN5.metrics_names} of {scores}')

  scores_training = ANN5.evaluate(X_train, y_train, verbose=0)
  print(f'Training Score for fold {fold_no}: {ANN5.metrics_names} of {scores_training}')

  loss_per_fold.append(scores)
  Train_loss_per_fold.append(scores_training)

## convert data of this fold to binary
  true_outliers_bi = y_test_array.copy()
  predicted_outliers_bi = test_predictions.copy()

  for i in range(0,len(true_outliers_bi)):
      if true_outliers_bi[i] > 3:
          true_outliers_bi[i] = 1
      else:
          true_outliers_bi[i] = 0

  for i in range(0,len(predicted_outliers_bi)):
      if predicted_outliers_bi[i] > 3:
          predicted_outliers_bi[i] = 1
      else:
          predicted_outliers_bi[i] = 0

## ROC
  fpr["fpr%s" %fold_no], tpr["tpr%s" %fold_no],threshold["threshold%s" %fold_no] = roc_curve(true_outliers_bi, predicted_outliers_bi)
  roc_auc["roc_auc%s" %fold_no] = auc(fpr["fpr%s" %fold_no], tpr["tpr%s" %fold_no])

  tprs.append(tpr["tpr%s" %fold_no])
  fprs.append(["fpr%s" %fold_no])
  aucs.append(["roc_auc%s" %fold_no])

## PR
  lr_precision["lr_precision%s" %fold_no], lr_recall["lr_recall%s" %fold_no], _ = precision_recall_curve(true_outliers_bi, predicted_outliers_bi)
  lr_f1["lr_f1%s" %fold_no], lr_auc["lr_auc%s" %fold_no] = f1_score(true_outliers_bi, predicted_outliers_bi), auc(lr_recall["lr_recall%s" %fold_no], lr_precision["lr_precision%s" %fold_no])

  # Increase fold number
  fold_no = fold_no + 1

#ROC plots
plt.title('Receiver Operating Characteristic')
plt.plot(fpr['fpr1'], tpr['tpr1'], linestyle='--',color='green', label='fold_No=1'+',AUC = %0.2f' % roc_auc['roc_auc1'])
plt.plot(fpr['fpr2'], tpr['tpr2'], linestyle='--',color='blue', label='fold_No=2'+',AUC = %0.2f' % roc_auc['roc_auc2'])
plt.plot(fpr['fpr3'], tpr['tpr3'], linestyle='--',color='red', label='fold_No=3'+',AUC = %0.2f' % roc_auc['roc_auc3'])
plt.plot(fpr['fpr4'], tpr['tpr4'], linestyle='--',color='black', label='fold_No=4'+',AUC = %0.2f' % roc_auc['roc_auc4'])
plt.plot(fpr['fpr5'], tpr['tpr5'], linestyle='--',color='yellow', label='fold_No=5'+',AUC = %0.2f' % roc_auc['roc_auc5'])
plt.plot(fpr['fpr6'], tpr['tpr6'], linestyle='-',color='green', label='fold_No=6'+',AUC = %0.2f' % roc_auc['roc_auc6'])
plt.plot(fpr['fpr7'], tpr['tpr7'], linestyle='-',color='blue', label='fold_No=7'+',AUC = %0.2f' % roc_auc['roc_auc7'])
plt.plot(fpr['fpr8'], tpr['tpr8'], linestyle='-',color='red', label='fold_No=8'+',AUC = %0.2f' % roc_auc['roc_auc8'])
plt.plot(fpr['fpr9'], tpr['tpr9'], linestyle='-',color='black', label='fold_No=9'+',AUC = %0.2f' % roc_auc['roc_auc9'])
plt.plot(fpr['fpr10'], tpr['tpr10'], linestyle='-',color='yellow', label='fold_No=10'+',AUC = %0.2f' % roc_auc['roc_auc10'])

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle='-',color='orange')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#PR plots
plt.title('PR plot')
plt.plot(lr_recall['lr_recall1'], lr_precision['lr_precision1'], linestyle='--',color='green', label='fold_No=1'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc1'], lr_f1['lr_f11']))
plt.plot(lr_recall['lr_recall2'], lr_precision['lr_precision2'], linestyle='--',color='blue', label='fold_No=2'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc2'], lr_f1['lr_f12']))
plt.plot(lr_recall['lr_recall3'], lr_precision['lr_precision3'], linestyle='--',color='red', label='fold_No=2'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc3'], lr_f1['lr_f13']))
plt.plot(lr_recall['lr_recall4'], lr_precision['lr_precision4'], linestyle='--',color='black', label='fold_No=4'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc4'], lr_f1['lr_f14']))
plt.plot(lr_recall['lr_recall5'], lr_precision['lr_precision5'], linestyle='--',color='yellow', label='fold_No=5'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc5'], lr_f1['lr_f15']))
plt.plot(lr_recall['lr_recall6'], lr_precision['lr_precision6'], linestyle='-',color='green', label='fold_No=6'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc6'], lr_f1['lr_f16']))
plt.plot(lr_recall['lr_recall7'], lr_precision['lr_precision7'], linestyle='-',color='blue', label='fold_No=7'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc7'], lr_f1['lr_f17']))
plt.plot(lr_recall['lr_recall8'], lr_precision['lr_precision8'], linestyle='-',color='red', label='fold_No=8'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc8'], lr_f1['lr_f18']))
plt.plot(lr_recall['lr_recall9'], lr_precision['lr_precision9'], linestyle='-',color='black', label='fold_No=9'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc9'], lr_f1['lr_f19']))
plt.plot(lr_recall['lr_recall10'], lr_precision['lr_precision10'], linestyle='-',color='yellow', label='fold_No=10'+',AUC = %0.2f f1=%.2f' % (lr_auc['lr_auc10'], lr_f1['lr_f110']))
plt.legend(loc = 'lower left')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# global plot true vs. predicted

#a = plt.axes(aspect='equal')
#plt.scatter(predcited_y, true_y)
#plt.xlabel('Predictions [residual]')
#plt.ylabel('True Values [residual]')
#lims = [-5, 20]
#plt.xlim(lims)
#plt.ylim(lims)
#prediction_plot = plt.plot(lims, lims)

