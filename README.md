# DeepCarc

### We made a simple version for the proposed DeepCarc model (Mol2vec + supervised selection method) under the folder of deepcarc. In this simple version, all you need is to update the path in the main.py file for repeating the result in the manuscript. 

### In the DeepCarc project, we explored three chemical descriptors (Mol2vec, Mold2 and MACCS) to develop the base classifiers. Therefore, the corresponding data and script were included under the descriptor folder. 

## mol2vec
- Data 
  - carcinogenecity_mol2vec_297.csv (input for base classifiers)
  - mol2vec_supervised.csv (selected base classifiers from the supervised selection method)
  - mol2vec_org.csv (selected base classifiers from the original selection method)
  - mol2vec_supervised_weights.h5 (the developed DeepCarc model from mol2vec and supervised selection method)
  - mol2vec_org_weights.h5 (the developed DeepCarc model from mol2vec and original selection method)
  
- Scripts
  - base_knn.py (develop base classifiers)
  - base_lr.py (develop base classifiers)
  - base_svm.py (develop base classifiers)
  - base_rf.py (develop base classifiers)
  - base_xgboost.py (develop base classifiers)  
  - validation_predictions_combine.py (combine predicted probabilities for development set)
  - test_predictions_combine.py (combine predicted probabilities for test set)  
  - mol2vec_supervised.py (Metaclass of the DeepCarc model with Mol2vec and the supervised selection strategy)
  - mol2vec_org.py (Metaclass of the DeepCarc model with Mol2vec and the original selection strategy)  
  
- Notes
  Please update all the path in the scripts with your local path and follow the script order list above to repeat the result.
  
## mold2
- Follow the same logic as mol2vec

## maccs
- Follow the same logic as mol2vec  
