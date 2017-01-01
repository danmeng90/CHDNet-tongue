#coding=utf-8
from __future__ import division
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import random
import math
from sklearn.metrics import roc_curve
import time

X = np.genfromtxt('../CHDNet_tongue/feature_random.csv', delimiter=',')
y = np.genfromtxt('../CHDNet_tongue/label_random.csv', delimiter=',')

#================================================

#=================================================

def evulate(ytest,pred):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(ytest)):
        if pred[i]>0.5:
            pred[i] = 1
        else:
            pred[i] = 0
    for i in range(len(ytest)):
        if ytest[i]==1 and pred[i]==1:
            TP = TP + 1
        elif ytest[i]==1 and pred[i]==0:
            FN = FN + 1
        elif ytest[i]==0 and pred[i]==1:
            FP  = FP + 1
        else:
            TN = TN + 1
    return TP,FN,FP,TN
def assement(TP,FN,FP,TN):
    correct_rate = (TP + TN)/(TP + FN + FP + TN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    ppv = TP/(TP+FP)
    if TN + FN == 0:
        npv = 0
    else:
        npv = TN/(TN+FN)

    F1_score = 2*TP/(2*TP+FP+FN)
    return correct_rate,sensitivity,specificity,ppv,npv,F1_score

CR = []
SE = []
SP = []
T = []
F1 = []
PPV = []
NPV = []

for j in range(0,10):

    total_correct_rate = 0
    total_sensitivity = 0
    total_specificity = 0
    total_F1_score = 0
    total_time = 0
    total_ppv = 0
    total_npv = 0

    skf = StratifiedKFold(y, 5, shuffle= True)
    for train_index, test_index in skf:
        start_time = time.time()
        train_data = X[train_index];train_y = y[train_index]
        test_data = X[test_index];test_y = y[test_index]
        clf = RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=1,random_state=0)
        clf.fit(train_data, train_y)
        pred = clf.predict(test_data)
        TP,FN,FP,TN = evulate(test_y,pred)
        # print TP,FN,FP,TN
        correct_rate, sensitivity, specificity, ppv, npv, F1_score = assement(TP, FN, FP, TN)
        end_time = time.time()
        # print "correct_rate",correct_rate
        # print "sensitivity",sensitivity
        # print "specificity",specificity
        # print "F1_score:",F1_score
        # print "runing time:",end_time-start_time
        total_correct_rate = total_correct_rate + correct_rate
        total_sensitivity = total_sensitivity + sensitivity
        total_specificity = total_specificity + specificity
        total_F1_score = total_F1_score + F1_score
        total_ppv = total_ppv + ppv
        total_npv = total_npv + npv
        total_time = total_time + end_time - start_time

        # print "avgCorrect",total_correct_rate/5
        # print "avgSensitivity",total_sensitivity/5
        # print "avgSpecificity",total_specificity/5
        # print "avgF1_score",total_F1_score/5
        # print "avgPPV------------------>",total_ppv/5
        # print "avgNPV-------------------->",total_NPV/5
        # print "avgRuning time",total_time/5

    CR.append(total_correct_rate / 5)
    SE.append(total_sensitivity / 5)
    SP.append(total_specificity / 5)
    T.append(total_time / 5)
    F1.append(total_F1_score / 5)
    PPV.append(total_ppv / 5)
    NPV.append(total_npv / 5)

import numpy as np

print 'CR',np.mean(CR)
print 'SE', np.mean(SE)
print 'SP', np.mean(SP)
print 'PPV',np.mean(PPV)
print 'NPV',np.mean(NPV)
print 'F1',np.mean(F1)
# print 'T',np.mean(T)

# print 'CR_VAR',np.var(CR)
# print 'SE_VAR',np.var(SE)
# print 'SP_VAR',np.var(SP)
# print 'F1_VAR',np.var(F1)
# print 'T_VAR',np.var(T)
# print 'PPV_VAR',np.var(PPV)
# print 'NPV_VAR',np.var(NPV)
