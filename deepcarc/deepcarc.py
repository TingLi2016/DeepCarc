#!/account/tli/anaconda3/bin/python

import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

### Set a seed value
seed_value = 11
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

### Fix float type
from keras import backend as K
K.set_floatx('float32')
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

### Import packages
import itertools
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,balanced_accuracy_score
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras import optimizers
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.layers import Dense, Activation, BatchNormalization, GaussianNoise, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


def model_predict(X, y, model, col_name):
    y_pred = model.predict(X)
    y_pred_class = np.where(y_pred > 0.5, 1, 0)
    pred_result = pd.DataFrame()
    pred_result['id'] = y.index
    pred_result['y_true'] = y.values
    pred_result['prob_'+col_name] = y_pred
    pred_result['class_'+col_name] = y_pred_class

    result = measurements(y, y_pred_class, y_pred)
    return pred_result, result

def measurements(y_test, y_pred, y_pred_prob):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob) 
    sensitivity = metrics.recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    npv = TN/(TN+FN)
    return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy]
    
def main(df, test_df, model_path, name):
    
    X = df.iloc[:, 3:]
    print(X.shape)
    y = df.loc[:, 'y_true']
    X_test = test_df.iloc[:, 3:]
    print(X_test.shape)
    y_test = test_df.loc[:, 'y_true']

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_test = sc.transform(X_test)    
    

    ### load the model
    best_model = load_model(model_path)

    
    ### predict test set
    test_class, test_result = model_predict(X_test, y_test, best_model, name)
    train_class, train_result= model_predict(X, y, best_model, name)


    K.clear_session()
    tf.reset_default_graph() 
    return test_class, test_result, train_class, train_result 


def sep_performance(df):
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC', 'BA']
    for i, col in enumerate(cols):
        if i == 0:
            df[col] = df.value.str.split(',').str[i].str.split('[').str[1].values
        elif i == len(cols)-1:
            df[col] = df.value.str.split(',').str[i].str.split(']').str[0].values
        else:
            df[col] = df.value.str.split(',').str[i].values

    for i, col in enumerate(cols):
        if i < 4:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
            df[col] = round(df[col], 3)
    del df['value']
            
    return df

def reform_result(results):
    df = pd.DataFrame(data=results.items())
    ###reform the result data format into single colum
    df = df.rename(columns={0:'name', 1:'value'})
    df['name'] = df['name'].astype('str')
    df['value'] = df['value'].astype('str')
    df = sep_performance(df)
    return df

def deepcarc_prediction(probability_path, name, model_path, result_path):

    ### Import data
    data = pd.read_csv(probability_path+'/validation_probabilities_' + name + '.csv')
    test = pd.read_csv(probability_path+'/test_probabilities_' + name + '.csv')

    ### Set path
    path2 = result_path + '/validation_class'
    path3 = result_path + '/validation_performance'
    path4 = result_path + '/test_class'
    path5 = result_path + '/test_performance'


    ### Initial performance dictionary
    test_results={}
    train_results={}

    ### Get the prediction and save results
    test_class, test_result, train_class, train_result  = main(data, test, model_path, name)

    test_results[name]=test_result
    test_class.to_csv(path4+'/test_'+name+'.csv')

    train_results[name]=train_result
    train_class.to_csv(path2+'/validation_'+name+'.csv')

    reform_result(test_results).to_csv(path5+'/dnn_test_' + name +'.csv')
    reform_result(train_results).to_csv(path3+'/dnn_validation_' + name +'.csv')
    print('Prediction is completed')

print("--- %s seconds ---" % (time.time() - start_time))
