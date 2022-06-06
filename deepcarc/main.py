#!/account/tli/anaconda3/bin/python

import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

### import scripts
import base_knn
import base_lr
import base_svm
import base_rf
import base_xgboost

import validation_predictions_combine
import test_predictions_combine

import deepcarc

### define the path for data, base classifers results, 
data = pd.read_csv('/account/tli/carcinogenecity/script/mol2vec/mol2vec_github_three/carcinogenecity_mol2vec_297.csv', low_memory=False)# path for carcinogenecity_mol2vec_297.csv
data_split = pd.read_csv('/account/tli/carcinogenecity/script/mol2vec/mol2vec_github_three/data_split.csv')# path for mol2vec_github_two/data_split.csv
mcc = pd.read_csv('/account/tli/carcinogenecity/script/mol2vec/mol2vec_github_three/mol2vec_supervised.csv') # path for mol2vec_supervised.csv

base_path = '/account/tli/carcinogenecity/results/mol2vec/para_selection' # path for base classifiers
probability_path = '/account/tli/carcinogenecity/results/mol2vec/para_selection/probabilities_output' # path for the combined probabilities (model-level representations)
name = 'supervised' # can be any name 
model_path = '/account/tli/carcinogenecity/script/mol2vec/mol2vec_github/mol2vec_supervised_weights.h5' # path for mol2vec_supervised_weights.h5
result_path = '/account/tli/carcinogenecity/results/mol2vec/para_selection/result' # path for the final deepcarc predictions

### run the scripts
base_knn.generate_baseClassifiers(data, data_split, mcc, base_path+'/knn')
base_lr.generate_baseClassifiers(data, data_split, mcc, base_path+'/lr')
base_svm.generate_baseClassifiers(data, data_split, mcc, base_path+'/svm')
base_rf.generate_baseClassifiers(data, data_split, mcc, base_path+'/rf')
base_xgboost.generate_baseClassifiers(data, data_split, mcc, base_path+'/xgboost')

validation_predictions_combine.combine_validation_probabilities(base_path, mcc, probability_path, name)
test_predictions_combine.combine_test_probabilities(base_path, mcc, probability_path, name)

deepcarc.deepcarc_prediction(probability_path, name, model_path, result_path)


print("--- %s seconds ---" % (time.time() - start_time))

