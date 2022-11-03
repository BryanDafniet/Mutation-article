#Packages

# import the necessary packages
import re
import sys
from tabulate import tabulate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier
#from scikeras.wrappers import KerasClassifier
#from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval

import multiprocessing

# function to get unique values
def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
    # print list
    


#File opening

All_Data = pd.read_csv("All_DB.csv", sep=",")
print(All_Data.shape)
print(All_Data.head)


# File cleaning
Comp_wSoc=All_Data.iloc[:, [1, 0, 9,10]]
Inchi_Unique=unique(Comp_wSoc['InchiKey'])
#rsID_Unique=unique(Comp_wSoc['rsID'])
Keys_dico=['InchiKey', 'SOC_abbrev']
# Using Dictionary comprehension
Abbrev_list = {key: [] for key in Keys_dico}
#print(Abbrev_list)
#print(Inchi_Unique)
for inchi in Inchi_Unique :
    #print(inchi)
    tmp=Comp_wSoc.loc[Comp_wSoc['InchiKey']==inchi]
    #print(unique(tmp['SOC_abbrev']))
    Abbrev_list['InchiKey'].append(inchi)
    Abbrev_list['SOC_abbrev'].append(unique(tmp['SOC_abbrev']))
#print(Abbrev_list)   

# Final transformation
df_Abbrev_list=pd.DataFrame(Abbrev_list)
#print(df_Abbrev_list)
print(Comp_wSoc)

def matrix_creation(mat_type): #Matrix creation, either with target interactions with 0/1s or with the frequency of the mutations.
    column_names=unique(All_Data['rsID'])
    df_mut=pd.DataFrame(0,columns=column_names, index=df_Abbrev_list['InchiKey'])
    if mat_type == 'frequency' or mat_type == 'Frequency':
        for inchi in Inchi_Unique :
            tmp_rows=Comp_wSoc.loc[Comp_wSoc['InchiKey']==inchi]
            #print(inchi)
            tmp_rsID=unique(All_Data.loc[All_Data.InchiKey == inchi, 'rsID'])
            rsID_Unique=unique(tmp_rsID)
            #print(rsID_Unique)
            columns=All_Data.columns
            rows=df_mut.index
            for rsID in rsID_Unique : #Have some same mutations w/ different frequencies, selecting the highest one
                rank_freq=All_Data.loc[(All_Data['rsID'] == rsID) & (All_Data['InchiKey'] == inchi),'rs_Freq']
                rank_freq=rank_freq.sort_values(ascending=True).iloc[0]

    if mat_type == 'target' or mat_type == 'Target': # Modifying dataframe with 1s
        for inchi in Inchi_Unique :
            tmp_rows=Comp_wSoc.loc[Comp_wSoc['InchiKey']==inchi]
            #print(tmp_rows)
            tmp_rsID=unique(All_Data.loc[All_Data.InchiKey == inchi, 'rsID'])
            #print(tmp_rsID)
            df_mut.loc[df_mut.index == inchi, tmp_rsID] = 1
    return df_mut

def Soc_sel(SOCS, df_ToUse): #Adding a column for every SOC with 0 or 1 if the compound has an ADR belonging to a SOC
    for SOC in SOCS:
        Keys_dico=['InchiKey', SOC]
        list_OneSoc = {key: [] for key in Keys_dico}
        #print(Abbrev_list)
        #print(Inchi_Unique)
        for inchi in Inchi_Unique :
            tmp=All_Data.loc[All_Data['InchiKey']==inchi]
            #print(tmp['SOC_abbrev'].values)
            list_OneSoc['InchiKey'].append(inchi)
            if SOC in tmp['SOC_abbrev'].values :
                list_OneSoc[SOC].append(1)
            else :

                list_OneSoc[SOC].append(0)

            # Final transfo
        df_list_OneSoc=pd.DataFrame(list_OneSoc)

        if count == 1:
            df_ToUse.reset_index(inplace=True) #Uncomment when running it the first time
            df_All_Merged=df_ToUse.merge(df_list_OneSoc[['InchiKey',SOC]])
            count += 1
        else:
            df_All_Merged=pd.merge(df_All_Merged,df_list_OneSoc, on='InchiKey', how = 'left')

        df_All_Merged.set_index('InchiKey',inplace = True)
        df_All_Merged.iloc[:,0:]=df_All_Merged.iloc[:,0:].astype(np.int32) #comment/uncomment wether it is frequency/target to make sure the good datatype is created
        #df_All_Merged.iloc[:,0:]=df_All_Merged.iloc[:,0:].astype(np.float32)
        df_All_Merged[SOC] = np.array(df_All_Merged[SOC])
        print("Done with ",SOC)
    print(df_All_Merged.head)
    return df_All_Merged, df_list_OneSoc

#### Try to remove mutations with only a specific number mol
def threshold(df_All_Merged,type_threshold, number):
    NoMol_List=[]
    if type_threshold == 'less' or type_threshold == 'Less':
        df_OneMol=df_All_Merged.copy()
        for column in df_OneMol.columns[:2921]:   
            # Select column contents by column
            # name using [] operator
            columnSeriesObj = df_OneMol[column]
            List_toCheck=columnSeriesObj.value_counts().tolist()
            #print(List_toCheck)
            if List_toCheck[0] == 1030:
                print(column)
                NoMol_List.append(column)
                del df_OneMol[column]
                continue
            if len(df_OneMol)-List_toCheck[0] <= number :
                del df_OneMol[column]
        return df_OneMol
    if type_threshold == 'greater' or type_threshold == 'Greater':
        df_MultipleMol=df_All_Merged.copy()
        for column in df_MultipleMol.columns[:2921]:   
            # Select column contents by column
            # name using [] operator
            columnSeriesObj = df_MultipleMol[column]
            List_toCheck=columnSeriesObj.value_counts().tolist()
            print(List_toCheck)
            if List_toCheck[0] == 1030:
                print(column)
                NoMol_List.append(column)
                del df_MultipleMol[column]
                continue
            if len(df_MultipleMol)-List_toCheck[0] > number :
                del df_MultipleMol[column]
            if (List_toCheck[0] == len(df_MultipleMol)) or  (List_toCheck[1] == len(df_MultipleMol)):
                print(column)
                NoMol_List.append(column)
                del df_Null[column]
        return df_MultipleMol
    if type_threshold == "Null" or type_threshold == "null":
        df_Null=df_All_Merged.copy()
        for column in df_Null.columns[:2921]:   
            columnSeriesObj = df_Null[column]
            List_toCheck=columnSeriesObj.value_counts().tolist()
            if (List_toCheck[0] == len(df_Null)) or  (List_toCheck[1] == len(df_Null)):
                print(column)
                NoMol_List.append(column)
                del df_Null[column]
        return df_Null

def creation_set(df_All_Merged, SOC):#Creation of training and test sets
    #mlb = MultiLabelBinarizer()
    X=df_All_Merged.iloc[:,0:2894] #Number of columns - number of soc -1 (starts at 0) - null columns values
    Y = pd.DataFrame(df_All_Merged[SOC])

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, shuffle = True, random_state = 1234)


    return train_x, train_y, test_x,test_y

def make_model(batch_size, nb_epoch): #MNN models
    model = Sequential()
    #model.add(Dense(512, activation='relu', input_shape= (2894,))) # Change input shape number based on length of column
    #model.add(keras.layers.Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(Dense(128, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(Dense(64, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    #model.add(keras.layers.Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def tuning(SOCS, df_All_Merged): #Tuning the best neural network parameters and running them for every SOC
    # Seed value (can actually be different for each attribution step)
    seed_value= 2

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    from numpy.random import seed
    seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    import itertools

    data_pred=[]
    Compound_pred_train=pd.DataFrame(columns=['SOC','X_InchiKey','Y_true_train', 'Y_pred_train'])
    Compound_pred_test=pd.DataFrame(columns=['SOC','X_InchiKey','Y_true_test', 'Y_pred_test'])


    for SOC in SOCS:
        X_train,Y_train,X_test,Y_test=creation_set(df_All_Merged, SOC)  #Training/Val/Test set creation
        tmp_list=[]
        tmp_list.append(SOC)
        #print(X_train.head)
        #print(Y_train.head)
        #print(X_test.head)
        #print(Y_test.head)
        num_cores=multiprocessing.cpu_count()
        #print(np.unique(Y_train))
        #print(type(Y_train))
        manual_weight= {0: 1.0, 1:1000}
        # Count samples per class
        classes_zero = Y_train[Y_train[SOC] == 0]
        classes_one = Y_train[Y_train[SOC] == 1]
        # Convert parts into NumPy arrays for weight computation
        zero_numpy = classes_zero[SOC].to_numpy()
        one_numpy = classes_one[SOC].to_numpy()

        Y_train_weights = np.concatenate((zero_numpy,one_numpy))
        #print(Y_train)
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(Y_train), 
                                                          y=Y_train_weights)
        class_weights = dict(enumerate(class_weights))
        print(class_weights)
        batch_size = [5, 10, 20, 32, 64]
        epochs = [50,100,150,200]
        params_grid = dict(batch_size=batch_size, nb_epoch=epochs)
        k_model = KerasClassifier(build_fn=make_model,verbose = 0)
        #print("toto1")
        clf = model_selection.GridSearchCV(estimator=k_model, param_grid=params_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1234),
                       scoring='balanced_accuracy', return_train_score=True, verbose=0)
        #clf.cv_results_
        #print("toto2")
        clf.fit(X_train,Y_train, class_weight=class_weights)
        #print("toto3")


        y_true_train, y_pred_train=Y_train,clf.predict(X_train)
        print("Balanced Accuracy score for training set prediction %s" %balanced_accuracy_score(y_true_train, y_pred_train))
        tn_train, fp_train, fn_train, tp_train =confusion_matrix(y_true_train, y_pred_train).ravel()
        sensitivity_train = tp_train/(tp_train+fn_train)
        specificity_train=tn_train/(tn_train+fp_train)
        BA_train=balanced_accuracy_score(y_true_train, y_pred_train)
        tmp_list.extend((BA_train, sensitivity_train,specificity_train,clf.cv_results_['params'][clf.best_index_]))
        print(clf.cv_results_['params'][clf.best_index_])


        print("Best parameters set found on development set:")
        #print(Y_test)
        print(clf.best_params_)
        print()
        print(clf.best_score_)
        print()
        print("Grid scores on development set:")
        print()
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = Y_test, clf.predict(X_test)
        tn, fp, fn, tp =confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp/(tp+fn)
        specificity=tn/(tn+fp)
        BA=balanced_accuracy_score(y_true, y_pred)
        
        SOC_train=list(itertools.repeat(SOC, len(y_true_train)))
        SOC_test=list(itertools.repeat(SOC, len(y_true)))
        tmp_list.extend((BA, sensitivity,specificity, class_weights))
        print(Y_train.iloc[:, 0])
        df_train = pd.DataFrame(columns=['SOC','X_InchiKey','Y_true_train', 'Y_pred_train'])
        df_test = pd.DataFrame( columns=['SOC','X_InchiKey','Y_true_test', 'Y_pred_test'])
        df_train['SOC']=SOC_train
        df_train['X_InchiKey']=X_train.index.values
        df_train['Y_true_train']=list(Y_train.iloc[:, 0])
        df_train['Y_pred_train']=y_pred_train
        #print(df_train)
        df_test['SOC']=SOC_test
        df_test['X_InchiKey']=X_test.index.values
        df_test['Y_true_test']=list(Y_test.iloc[:, 0])
        df_test['Y_pred_test']=y_pred
        #print(df_test)
        Compound_pred_train=Compound_pred_train.append(df_train)
        Compound_pred_test=Compound_pred_test.append(df_test)
        print(Compound_pred_train)
        data_pred.append(tmp_list)
        print(Compound_pred_train)
        print()
        print(Compound_pred_test)

        print()
        print("Balanced Accuracy score %s" %balanced_accuracy_score(y_true, y_pred))
        #print(confusion_matrix(y_true, y_pred))
        print("Classification report  \n %s" %(classification_report(y_true, y_pred)))

        #print(y_true)
        #print(y_pred)    
        print()
    return data_pred,Compound_pred_train,Compound_pred_test
    #return Compound_pred_train,Compound_pred_test


df_mut = matrix_creation('Target') #Target or Frequencies
df_mut = df_mut.drop(df_mut.index[400]) #NA inchikey
df_All_Merged, df_list_OneSoc = Soc_sel(SOCS = SOC_list, df_ToUse=df_mut) # Creation of the SOC column and merge with the previous matrix
df_All_Merged_filter=threshold(df_All_Merged,'Null', 0)
NN_model, Data_train, Data_test=tuning(SOCS = SOC_list, df_All_Merged = df_All_Merged_filter) # Neural network function

content2=tabulate(NN_model, headers=["SOC","BA_train", "Sens", "Spe","Best_Params","BA","Sens","Spe","Class_weight"], tablefmt="tsv", floatfmt=".2f")
Content_train=tabulate(Data_train, headers=["SOC", "Y_true_train","Y_pred_train"], tablefmt="tsv", floatfmt=".2f")
Content_test=tabulate(Data_test, headers=["SOC", "Y_true_test","Y_pred_test"], tablefmt="tsv", floatfmt=".2f")
text_file=open("NN_Model_Target_Start_16_0_Dropout_1.csv","w")
text_file.write(content2)
text_file.close()
text_file=open("Training_Model_Target_Start_16_0_Dropout.csv","w")
text_file.write(Content_train)
text_file.close()
text_file=open("Test_Model_Target_Start_16_0_Dropout.csv","w")
text_file.write(Content_test)
text_file.close()


