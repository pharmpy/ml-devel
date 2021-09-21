#!/usr/bin/env python
# coding: utf-8

# In[Load Packages]:
# Basic code for Artificial Neural Network for tensorflow lite
# September 2021
# adapted from Alzahra Hamdan (August 2021) 

# load tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, initializers, Sequential
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

from json import JSONEncoder
# Make numpy printouts easier to read.
#import tflite_runtime.interpreter as tflite

np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)


# In[Define StratifiedKFold ]:


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


# In[Load dataset]:


#load data
raw_dataset = pd.read_csv('path/merged_datasets_for_simeval.csv')
dataset = raw_dataset.copy()
dataset.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model'], axis = 1, inplace=True)
dataset.head()

#split features and labels
x = dataset.copy()
y = x.pop('residual')
X = x.values

# In[Train Model ]:

#ANN5: This is the final model
# number of splits/folds
n_splits = 10                #number of folds
loss_per_fold = []           #to store test loss value in each fold
Train_loss_per_fold = []     #to store training loss value in each fold
predicted_y = np.array([])   #to store predicted residual value from each CV fold
predicted_prob = np.array([])
true_y = np.array([])        #to store true residual value from each CV fold
lr = 0.00007

# cross validated stratification
cv_stratified = StratifiedKFoldReg(n_splits=n_splits, shuffle=True, random_state=10)   

fold_no = 1
for ii, (train_index, test_index) in enumerate(cv_stratified.split(X, y)):
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test = X[train_index], X[test_index]

    #Define and summarize the model
    inps = layers.Input(shape=X_train[0].shape, dtype='float32')
    norm_layer = layers.Normalization(axis=1)
    norm_layer.adapt(X_train)
    x = norm_layer(inps)
    x = layers.Dense(48, activation='relu')(x)
    x = layers.Dense(24, activation='relu')(x)
    x = layers.Dense(12, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(1)(x)
    #pred_prob = layers.Dense(1, activation = 'softmax')(x)
    #ANN5 = models.Model(inputs=inps, outputs=(preds, pred_prob))
    ANN5 = models.Model(inputs=inps, outputs=preds)

    #Compile the model
    ANN5.compile(optimizer=optimizers.RMSprop(learning_rate=lr),loss='mse')

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    test_labels = y_test.to_list()
    test_labels = [round(num, 2) for num in test_labels]
    print(test_labels)   #to have a look at the true residual values for test dataset

    # Fit data to model
    history = ANN5.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=0)
  
    #to store values for plotting global predicted vs. true residual values
    test_predictions = ANN5.predict(X_test).flatten()
    predicted_y = np.append(predicted_y, test_predictions)

    #test_prob = ANN5.predict(X_test('pred_prob')).flatten()
    #predicted_prob = np.append(predicted_prob, test_predictions('pred_prob'))
    
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


# In[Plot prediction]:
    
# a = plt.axes(aspect='equal')
# plt.scatter(predicted_y, true_y)
# plt.xlabel('Predictions [residual]')
# plt.ylabel('True Values [residual]')
# lims = [-5, 20]
# plt.xlim(lims)
# plt.ylim(lims)
# prediction_plot = plt.plot(lims, lims)


# In[Save model and convert to tflite]:
# Save model and convert to tflite

# tf.saved_model.save(ANN5,'path/ML_CDD_simeval/ml-devel/outlier')

# tflite_model = tf.lite.TFLiteConverter.from_saved_model('path/ML_CDD_simeval/ml-devel/outlier').convert()

# with open('path/ML_CDD_simeval/ml-devel/outlier/outliers.tflite', 'wb') as f:
#     f.write(tflite_model)

# convert directly
tflite_model = tf.lite.TFLiteConverter.from_keras_model(ANN5).convert()

with open('path/ML_CDD_simeval/ml-devel/outlier/outliers.tflite', 'wb') as f:
    f.write(tflite_model)   
    
# In[Test tensorflow model]:
ANN5.save('path/ML_CDD_simeval/outlier' )    

#%%
# read in tensorflow model and test
new_model = tf.keras.models.load_model('path/ML_CDD_simeval/outlier')

#%%
# test on data

#%%
raw_data = pd.read_csv('path/merged_datasets_for_simeval.csv')
rawdat1 = raw_data.copy()
is_test1 = rawdat1['Model_number']=='PRZrun4'
rawdat1 = rawdat1[is_test1]
rawdat1.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model', 'residual'], 
                 axis = 1, 
                 inplace = True)    
# print(rawdat1.head())
    
test1 = new_model.predict(rawdat1)
print(test1)
#rawdat2 = rawdat1.copy
#rawdat2 = rawdat1.astype(np.float32)

# print(rawdat1.head())

# read in tensorflow model and test 
#test2 = new_model.predict(rawdat2)
#print(test2)



#%%    

# # code from: https://pynative.com/python-serialize-numpy-ndarray-into-json/
# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

# # load trained outlier model
# model = tf.keras.models.load_model('path/ML_CDD_simeval/outlier')

# # set up empty arrays
# out_conf = ()
# out_wt = ()

# # get NN configuration and weights
# for layer in model.layers: 
#     print(f'config out: {layer.get_config()}')
#     print('------------------------------------------')
#     print(f'weights: {layer.get_weights()}')
#     print('------------------------------------------')
#     out_conf = np.append(out_conf, layer.get_config())
#     out_wt = np.append(out_wt, layer.get_weights())

# first_layer_weights = model.layers[0].get_weights()[0]
# print('first layer weights: {first_layer_weights}')

#%%

# convert to json and save    
# outlier_conf = {'array':out_conf}
# OutlierMod = json.dumps(outlier_conf, cls=NumpyArrayEncoder)
# with open("OutlierMod.json", "w") as json_file:
#     json_file.write(OutlierMod)
#     print('Saved model to disk')

# convert to json and save    
# out_wt = out_wt.astype(np.float32)

# outlier_wts = {'array':out_wt}
# OutlierWts = json.dumps(outlier_wts, cls=NumpyArrayEncoder)
# with open("OutlierWts.json", "w") as json_file:
#     json_file.write(OutlierWts)
#     print('Saved weights to disk')



# outlier_mod = out_conf.to_json()
# with open("outlier_mod.json", "w") as json_file:
#     json_file.write(outlier_mod)
#     print('Saved model to disk')

# outlier_wts = out_wt.to_json()
# with open("outlier_wts.json", "w") as json_file:
#     json_file.write(outlier_wts)
#     print('Saved weights to disk')
#%%

# # serialize model to JSON
# outlier_mod = model.to_json()
# with open("outlier_mod.json", "w") as json_file:
#     json_file.write(outlier_mod)
# # serialize weights to HDF5
# model.save_weights("outlier_mod_wt.h5")
# print("Saved model to disk")



# In[ ]:


# # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(loss_per_fold)):
#   print('------------------------------------------------------------------------')
#   print(f'> Fold {i+1} - Training Loss: {Train_loss_per_fold[i]} - Testing Loss: {loss_per_fold[i]} -')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Test Loss: {np.mean(loss_per_fold)}')
# print(f'> Training Loss: {np.mean(Train_loss_per_fold)}')
# print('------------------------------------------------------------------------')


# In[ ]:


# #permutation importance
# feature_name = ['Model_subjects', 'Model_observations',
#                 'Obsi_Obs_Subj', 'Covariate_relations', 'Max_cov', 'Max_CWRESi', 'Median_CWRESi',
#                 'Max_EBEij_omegaj', 'OFVRatio', 'mean_ETC_omega']

# r = permutation_importance(ANN5, X, y,
#                             n_repeats=30,
#                             random_state=0, scoring='neg_mean_squared_error')

# for i in r.importances_mean.argsort()[::-1]:
#      if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#          print(f"{feature_name[i]:<10}"
#                f"{r.importances_mean[i]:.3f}"
#                f" +/- {r.importances_std[i]:.3f}")


# # In[ ]:


# #sensitivity analysis: cut-off=3 for both true and predicted residual
# true_positives = 0
# false_positives = 0
# true_negative = 0
# false_negative = 0
# for i in range(0,len(true_y)):
#     if abs(true_y[i]) > 3 and abs(predcited_y[i]) > 3:
#         true_positives = true_positives + 1
#     if abs(true_y[i]) <= 3 and abs(predcited_y[i]) > 3:
#         false_positives = false_positives + 1
#     if abs(true_y[i]) <= 3 and abs(predcited_y[i]) <= 3:
#         true_negative = true_negative + 1
#     if abs(true_y[i]) > 3 and abs(predcited_y[i]) <= 3:
#         false_negative = false_negative + 1

# print(f'> TP: {true_positives}')
# print(f'> FP: {false_positives}')
# print(f'> TN: {true_negative}')
# print(f'> FN: {false_negative}')

# sensitivity = true_positives/(true_positives + false_negative)
# specificity = true_negative/(true_negative + false_positives)
# precision = true_positives/(true_positives + false_positives)

# print(f'> Sensitivity: {round(sensitivity, 3)}')
# print(f'> Specificity: {round(specificity, 3)}')
# print(f'> Precision: {round(precision, 3)}')


# In[ ]:


# #sensitivity analysis: cut-off=3 for true and 2.5 for predicted residual
# true_positives = 0
# false_positives = 0
# true_negative = 0
# false_negative = 0
# for i in range(0,len(true_y)):
#     if abs(true_y[i]) > 3 and abs(predcited_y[i]) > 2.5:
#         true_positives = true_positives+1
#     if abs(true_y[i]) <= 3 and abs(predcited_y[i]) > 2.5:
#         false_positives = false_positives + 1
#     if abs(true_y[i]) <= 3 and abs(predcited_y[i]) <= 2.5:
#         true_negative = true_negative+1
#     if abs(true_y[i]) > 3 and abs(predcited_y[i]) <= 2.5:
#         false_negative = false_negative + 1

# print(f'> TP: {true_positives}')
# print(f'> FP: {false_positives}')
# print(f'> TN: {true_negative}')
# print(f'> FN: {false_negative}')

# sensitivity = true_positives/(true_positives + false_negative)
# specificity = true_negative/(true_negative + false_positives)
# precision = true_positives/(true_positives + false_positives)

# print(f'> Sensitivity: {round(sensitivity, 3)}')
# print(f'> Specificity: {round(specificity, 3)}')
# print(f'> Precision: {round(precision, 3)}')

