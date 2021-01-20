import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import balanced_accuracy_score
#import plotly.graph_objects as go
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=True)
#import plotly.graph_objs as go
import os

def printDatasetInfo(X_train,y_train,X_test,y_test):
    print("Number transactions X_train dataset: ", X_train.shape)
    print("Number transactions y_train dataset: ", y_train.shape)
    print("Number transactions X_test dataset: ", X_test.shape)
    print("Number transactions y_test dataset: ", y_test.shape)

def printDatasetInfoAfter(X_train,y_train):
    print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("After OverSampling, counts of label '-1': {}".format(sum(y_train == -1)))
# -------------------------------------
def loadDatasetsCreditCard():
    data = pd.read_csv('..\data\imbalanced_data\creditcard.csv')
    # print(data.info())
    
    # normalise the amount column
    data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1))
    # drop Time and Amount columns as they are not relevant for prediction purpose
    data = data.drop(['Time', 'Amount'], axis=1)
    # as you can see there are 492 fraud transactions
    # print(data['Class'].value_counts())
    y = data['Class']
    X = data.drop('Class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # describes info about train and test set
    # printDatasetInfo(X_train,y_train,X_test,y_test)
    return X_train,y_train,X_test,y_test

def loadDatasets(data):
    data = pd.read_csv(data)
#    def evaluation_control(data):
#       evaluation = (data.n < 300) & (data.v < 1000 ) & (data.d < 50) & (data.e < 500000) & (data.t < 5000)
#      data['complexityEvaluation'] = pd.DataFrame(evaluation)
#       data['complexityEvaluation'] = ['Succesful' if evaluation == True else 'Redesign' for evaluation in data.complexityEvaluation]
#   evaluation_control(data)
    
    print(len(data))
    y = data[data.columns[-1]]
    X = data[data.columns[:-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=0)
    # printDatasetInfo(X_train,y_train,X_test,y_test)
    # kmeansCluster(X_train,y_train)
    return X_train,y_train,X_test,y_test
# ---------------------------------
def LogisticRegession(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LogisticRegression

    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    predictions=lr.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(lr, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()

def RandomForestClassifier(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=410, class_weight="balanced")
    clf.fit(X_train,y_train)
    predictions=clf.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(clf, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()
def GradientBoostingClassifier(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = clf = GradientBoostingClassifier(n_estimators = 300,learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(X_train,y_train)
    predictions=clf.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(clf, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()
def ForestClassifier(X_train,y_train,X_test,y_test):		
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    brf = BalancedRandomForestClassifier(n_estimators=500,max_depth=17, random_state=3)
    brf.fit(X_train, y_train)
    predictions=brf.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(brf, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()
def Bagging(X_train,y_train,X_test,y_test):		
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVR
    from imblearn.ensemble import BalancedBaggingClassifier
    from sklearn.ensemble import BaggingRegressor
    
    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),sampling_strategy='auto',replacement=False,random_state=0)
    bbc.fit(X_train, y_train) 
    predictions=bbc.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(bbc, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()

def Boosting(X_train,y_train,X_test,y_test):		
    from imblearn.ensemble import EasyEnsembleClassifier
	
    eec = EasyEnsembleClassifier(n_estimators=200)
    eec.fit(X_train, y_train) 
    predictions=eec.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(eec, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()
def Stacking(X_train,y_train,X_test,y_test):		
    from sklearn.ensemble import StackingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import StackingClassifier
	
    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))]
    stack = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(random_state=42))
    stack.fit(X_train, y_train)
    predictions=stack.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(stack, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()

	
def NaiveBayes(X_train,y_train,X_test,y_test):
    from sklearn.naive_bayes import GaussianNB

    clf=GaussianNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(clf, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()

def DecisionTree(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(tree, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()
def SVM(X_train,y_train,X_test,y_test):
    from sklearn.svm import SVC

    svm = SVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    clf_disp = plot_roc_curve(svm, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")   
    plt.show()
def KNN(X_train,y_train,X_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(classification_report(y_test, predictions))
# -----------------------------------
def algorithmApplySMOTE(X_train,y_train,X_test,y_test):
    from imblearn.over_sampling import SMOTE

    # print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    # print("Before OverSampling, counts of label '-1': {} \n".format(sum(y_train == -1)))
    sm=SMOTE(random_state=0)
    X_train_res,y_train_res=sm.fit_sample(X_train,y_train)
    # printDatasetInfoAfter(X_train_res, y_train_res)
    algorithmApply(X_train_res, y_train_res,X_test,y_test)
def algorithmApplyBagging(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.ensemble import BalancedBaggingClassifier
    
    #bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),sampling_strategy='auto',replacement=False,random_state=0)
    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
    X_train_res,y_train_res=bbc.fit(X_train, y_train)
    algorithmApply(X_train_res, y_train_res,X_test,y_test)
def algorithmApplyBoosting(X_train,y_train,X_test,y_test):
    from imblearn.ensemble import EasyEnsembleClassifier
    
    eec = EasyEnsembleClassifier(random_state=42)
    X_train_res,y_train_res=eec.fit(X_train, y_train) 
    algorithmApply(X_train_res, y_train_res,X_test,y_test)
def algorithmApplyNearMiss(X_train,y_train,X_test,y_test):
    from imblearn.under_sampling import NearMiss

    # print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 1)))
    # print("Before Undersampling, counts of label '-1': {} \n".format(sum(y_train == -1)))
    nr = NearMiss()
    print(X_train.shape)
    X_train_miss, y_train_miss = nr.fit_resample(X_train, y_train)
    #X_train_miss,y_train_miss=nr.fit_sample(X_train,y_train)
    #printDatasetInfoAfter(X_train_miss, y_train_miss)
    print(X_train_miss.shape)
    print(y_train_miss.shape)
    print(X_test.shape)
    print(y_test.shape)
    algorithmApply(X_train_miss, y_train_miss,X_test,y_test)

def algorithmApplyKmeansSMOTE(X_train,y_train,X_test,y_test):
    from kmeans_smote import KMeansSMOTE

    # print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    # print("Before OverSampling, counts of label '-1': {} \n".format(sum(y_train == -1)))
    # import SMOTE module from imblearn library
    # pip install imblearn (if you don't have imblearn in your system)
    kmeans_smote = KMeansSMOTE(
        kmeans_args={
             'n_clusters': 150
         },
         smote_args={
             'k_neighbors': 21
         }
    )
    X_train_res,y_train_res=kmeans_smote.fit_sample(X_train,y_train)
    printDatasetInfoAfter(X_train_res, y_train_res)
    algorithmApply(X_train_res, y_train_res,X_test,y_test)


def algorithmApply(X_train,y_train,X_test,y_test):
    # LogisticRegession(X_train, y_train, X_test, y_test)
    # RandomForestClassifier(X_train, y_train, X_test, y_test)
    # NaiveBayes(X_train, y_train, X_test, y_test)
    # DecisionTree(X_train, y_train, X_test, y_test)
    # SVM(X_train, y_train, X_test, y_test)
    # KNN(X_train, y_train, X_test, y_test)
    # GradientBoostingClassifier(X_train, y_train, X_test, y_test)
    # ForestClassifier(X_train,y_train,X_test,y_test)
    # Bagging(X_train,y_train,X_test,y_test)
    Boosting(X_train,y_train,X_test,y_test)
    # Stacking(X_train,y_train,X_test,y_test)
    # XGBClassifier(X_train,y_train,X_test,y_test)
# data = '..\data\imbalanced_data\oil.csv'
data = 'pc1.csv'
# data = '..\data\imbalanced_data\pima-indians-diabetes.csv'
# data = '..\data\imbalanced_data\letter_img.csv'
minority = 1
majority = -1
# X_train,y_train,X_test,y_test=loadDatasetsCreditCard()
X_train,y_train,X_test,y_test=loadDatasets(data)
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
printDatasetInfo(X_train,y_train,X_test,y_test)
printDatasetInfoAfter(X_train,y_train)
print('Algorithm without overshampling:')
algorithmApply(X_train,y_train,X_test,y_test)
#print('SMOTE:')
#algorithmApplySMOTE(X_train,y_train,X_test,y_test)
# print('logisticRegessionNearMiss:')
# algorithmApplyNearMiss(X_train,y_train,X_test,y_test)
#print('KmeansSMOTE:')
#algorithmApplyKmeansSMOTE(X_train,y_train,X_test,y_test)

#print('Bagging:')
#algorithmApplyBagging(X_train,y_train,X_test,y_test)
#print('Boosting:')
#algorithmApplyBoosting(X_train,y_train,X_test,y_test)




