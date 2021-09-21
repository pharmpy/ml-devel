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
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])

        # find each element of y to which label corresponds in the sorted
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label

        # put labels according to the given y order then
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])

        return super().split(X, y_labels, groups)


# In[Import dataset]:


#load data
raw_dataset = pd.read_csv('merged_datasets_for_simeval.csv')
dataset = raw_dataset.copy()
dataset.drop(['dofv', 'ID', 'Study_ID', 'Model_number', 'lin_model'], axis = 1, inplace=True)
dataset.head()

#split features and labels
x = dataset.copy()
Y = x.pop('residual')
X = x.values

# In[Create 3 layer ANN model]:

layer = tf.keras.layers.Normalization(axis=1)
layer.adapt(X)

def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=X.shape), 
    layer,
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
  ])

    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.00007), loss='mse')
    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# In[Run Kfold Cross validation ]:


# Set up empty vectors for outputs
loss_per_fold = []           #to store test loss value in each fold
Train_loss_per_fold = []     #to store training loss value in each fold
predicted_y = np.array([])   #to store predicted residual value from each CV fold
true_y = np.array([])        #to store true residual value from each CV fold

num_folds = 10   

# cross validated stratification to keep ratios same across each split
cv_stratified = StratifiedKFoldReg(n_splits=num_folds, shuffle=True, random_state=10)   

fold_no = 1
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

  
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    test_labels = Y_test.to_list()
    test_labels = [round(num, 2) for num in test_labels]
    print(test_labels)   #to have a look at the true residual values for test dataset

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

# In[Plot prediction]:
    
# a = plt.axes(aspect='equal')
# plt.scatter(predicted_y, true_y)
# plt.xlabel('Predictions [residual]')
# plt.ylabel('True Values [residual]')
# lims = [-5, 20]
# plt.xlim(lims)
# plt.ylim(lims)
# prediction_plot = plt.plot(lims, lims)

# In[permutation importance]:
feature_name = ['Model_subjects', 'Model_observations',
                'Obsi_Obs_Subj', 'Covariate_relations', 'Max_cov', 'Max_CWRESi', 'Median_CWRESi',
                'Max_EBEij_omegaj', 'OFVRatio', 'mean_ETC_omega']

r = permutation_importance(model, X, y,
                            n_repeats=30,
                            random_state=0, scoring='neg_mean_squared_error')

for i in r.importances_mean.argsort()[::-1]:
      if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
          print(f"{feature_name[i]:<10}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")

# In[Save model and convert to tflite]:
# Save model and convert to tflite

# tf.saved_model.save(ANN5,'path/ML_CDD_simeval/ml-devel/outlier')

# tflite_model = tf.lite.TFLiteConverter.from_saved_model('path/ML_CDD_simeval/ml-devel/outlier').convert()

# with open('path/ML_CDD_simeval/ml-devel/outlier/outliers.tflite', 'wb') as f:
#     f.write(tflite_model)

# convert directly
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()

with open('path/ML_CDD_simeval/ml-devel/outlier/outliers_test.tflite', 'wb') as f:
    f.write(tflite_model)   
    
# In[Test tensorflow model]:
model.save('ml-devel/outlier/SC_testing' )    

#%%
# read in tensorflow model and test
new_model = tf.keras.models.load_model('ml-devel/outlier/SC_testing')
new_model.summary()

#%%
raw_data = pd.read_csv('merged_datasets_for_simeval.csv')
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




