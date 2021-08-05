import pandas as pd
import numpy as np

import sys

#var=sys.argv[1]


import os
from os import listdir
from os.path import isfile, join
import itertools
from functools import reduce

base_path = '/account/tli/carcinogenecity/results/maccs/para_selection'

knn_base_path = base_path + '/knn/validation_class' 
lr_base_path = base_path +  '/lr/validation_class'
svm_base_path = base_path + '/svm/validation_class'
rf_base_path = base_path + '/rf/validation_class'
xgboost_base_path = base_path + '/xgboost/validation_class'

def combine_validation_probabilities(n):
    path='/account/tli/carcinogenecity/results/maccs/para_selection/probabilities_output'

    ###get the seed
    mcc=pd.read_csv('/account/tli/carcinogenecity/data/maccs/supervise/maccs_' + n + '.csv')

    seed_knn = mcc[mcc.model == 'knn'].seed.unique()
    seed_lr = mcc[mcc.model == 'lr'].seed.unique()
    seed_svm = mcc[mcc.model == 'svm'].seed.unique()
    seed_rf = mcc[mcc.model == 'rf'].seed.unique()
    seed_xgboost = mcc[mcc.model == 'xgboost'].seed.unique()


    print('knn: ', len(seed_knn))
    print('lr: ', len(seed_lr))
    print('svm: ', len(seed_svm))
    print('rf: ', len(seed_rf))
    print('xgboost: ', len(seed_xgboost))

    tmp = pd.read_csv(join(knn_base_path, 'validation_knn_paras_3_K_9.csv'))
    knn = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_knn):    
        col1 = [col for col in tmp.columns if 'prob_knn_seed_'+str(seed) in col]
        knn['knn_seed_'+str(seed)]=tmp[[*col1]]


    tmp = pd.read_csv(join(lr_base_path, 'validation_lr_paras_1.csv'))
    lr = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_lr):
        col1 = [col for col in tmp.columns if 'prob_lr_seed_'+str(seed) in col]
        lr['lr_seed_'+str(seed)]=tmp[[*col1]]


    tmp = pd.read_csv(join(svm_base_path, 'validation_svm_paras_5.csv'))
    svm = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_svm):
        col1 = [col for col in tmp.columns if 'prob_svm_seed_'+str(seed) in col]
        svm['svm_seed_'+str(seed)]=tmp[[*col1]]


    tmp = pd.read_csv(join(rf_base_path, 'validation_rf__paras_51.csv'))
    rf = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_rf):
        col1 = [col for col in tmp.columns if 'prob_rf_seed_'+str(seed) in col]
        rf['rf_seed_'+str(seed)]=tmp[[*col1]]


    tmp = pd.read_csv(join(xgboost_base_path, 'validation_xgboost_paras_87.csv'))
    xgboost = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_xgboost):
        col1 = [col for col in tmp.columns if 'prob_xgboost_seed_'+str(seed) in col]
        xgboost['xgboost_seed_'+str(seed)]=tmp[[*col1]]


    del lr['y_true']
    del svm['y_true']
    del rf['y_true']
    del xgboost['y_true']


    data = reduce(lambda x,y: pd.merge(x,y, on='id', how='left'), [knn, lr, svm, rf, xgboost])
    data.to_csv(join(path+'/validation_probabilities_' + str(n) + '.csv'))

for n in ['supervised', 'org']:
    combine_validation_probabilities(n)
#combine_validation_probabilities('supervised')