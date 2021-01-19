import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
from kmeans_smote import KMeansSMOTE

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # describes info about train and test set
    # printDatasetInfo(X_train,y_train,X_test,y_test)
    return X_train,y_train,X_test,y_test

def loadDatasets(data):
    data = pd.read_csv(data)
    # print(len(data))
    y = data[data.columns[-1]]
    X = data[data.columns[:-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
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

def RandomForestClassifier(X_train,y_train,X_test,y_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(random_state=0, class_weight="balanced")
    clf.fit(X_train,y_train)
    predictions=clf.predict(X_test)
    print(classification_report(y_test,predictions))

def NaiveBayes(X_train,y_train,X_test,y_test):
    from sklearn.naive_bayes import GaussianNB

    clf=GaussianNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(classification_report(y_test,predictions))

def DecisionTree(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    print(classification_report(y_test, predictions))

def SVM(X_train,y_train,X_test,y_test):
    from sklearn.svm import SVC

    svm = SVC()
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print(classification_report(y_test, predictions))

def KNN(X_train,y_train,X_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
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

def algorithmApplyNearMiss(X_train,y_train,X_test,y_test):
    from imblearn.under_sampling import NearMiss

    # print("Before Undersampling, counts of label '1': {}".format(sum(y_train == 1)))
    # print("Before Undersampling, counts of label '-1': {} \n".format(sum(y_train == -1)))
    nr = NearMiss()
    X_train_miss, y_train_miss = nr.fit_sample(X_train, y_train)

    X_train_miss,y_train_miss=nr.fit_sample(X_train,y_train)
    printDatasetInfoAfter(X_train_miss, y_train_miss)
    algorithmApply(X_train_miss, y_train_miss,X_test,y_test)

def algorithmApplyKmeansSMOTE(X_train,y_train,X_test,y_test):

    # print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    # print("Before OverSampling, counts of label '-1': {} \n".format(sum(y_train == -1)))
    # import SMOTE module from imblearn library
    # pip install imblearn (if you don't have imblearn in your system)
    kmeans_smote = KMeansSMOTE(
        # kmeans_args={
        #     'n_clusters': 10
        # },
        # smote_args={
        #     'k_neighbors': 10
        # }
    )
    X_train_res,y_train_res=kmeans_smote.fit_sample(X_train,y_train)
    # printDatasetInfoAfter(X_train_res, y_train_res)
    algorithmApply(X_train_res, y_train_res,X_test,y_test)
# -------------------------------------
def CIR_old(X_train,y_train,X_test,y_test, minority, majority):
    import random

    # Delete first column
    # del data['id']

    # Convert boolean data column to int
    # last_column = list(data.columns)[-1]
    # data[last_column] = data[last_column] * 1

    data = X_train
    data.insert(len(X_train.columns), y_train.name, y_train)
    Do = data.iloc[np.where(data[data.columns[-1]] == minority)]
    Dj = data.iloc[np.where(data[data.columns[-1]] == majority)]
    Do_X = Do[Do.columns[:-1]]
    C = np.mean(Do_X)
    dist = []
    for r in range(len(Do)):
        dist.append(np.linalg.norm(Do_X.iloc[r, :] - C))
    dist.sort()
    Dmin = dist[0]
    n = len(Dj) - len(Do)
    k = []
    for i in range(n):
        k.append(random.random())
    data_new = pd.DataFrame(columns=list(data.columns))
    for i in k:
        temp = Dmin + i * C
        data_new = data_new.append(temp, ignore_index=True)
    last_column_value = [minority] * n
    last_column_name = list(data.columns)[-1]
    data_new[last_column_name] = last_column_value
    Do_new = Do.append(data_new, ignore_index=True)
    D_total = Dj.append(Do_new, ignore_index=True)
    y_train = D_total[D_total.columns[-1]]
    X_train = D_total[D_total.columns[:-1]]
    algorithmApply(X_train, y_train, X_test, y_test)

def CIR(X_train,y_train,X_test,y_test, minority, majority):
    import random

    data = X_train
    data.insert(len(X_train.columns), y_train.name, y_train)
    Do = data.iloc[np.where(data[data.columns[-1]] == minority)]
    Dj = data.iloc[np.where(data[data.columns[-1]] == majority)]
    Do_X = Do[Do.columns[:-1]]

    # C_temp = []
    # for i in range(len(Do_X.columns)):
    #     C_temp.append(statistics.mode(Do_X.iloc[:, i]))
    # C = pd.Series(C_temp, index=list(Do_X.columns))
    C = np.mean(Do_X)
    dist = {}
    for r in range(len(Do)):
        dist[r]=np.linalg.norm(Do_X.iloc[r, :] - C)
    r_min = min(dist, key=dist.get)
    Dmin = Do_X.iloc[r_min, :]
    n = len(Dj) - len(Do)
    k = []
    for i in range(n):
        k.append(random.random())
    data_new = pd.DataFrame(columns=list(data.columns))
    for i in k:
        temp = Dmin + i * C
        data_new = data_new.append(temp, ignore_index=True)
    last_column_value = [minority] * n
    last_column_name = list(data.columns)[-1]
    data_new[last_column_name] = last_column_value
    Do_new = Do.append(data_new, ignore_index=True)
    D_total = Dj.append(Do_new, ignore_index=True)
    y_train = D_total[D_total.columns[-1]]
    X_train = D_total[D_total.columns[:-1]]
    algorithmApply(X_train, y_train, X_test, y_test)

def KmeansCIR(X_train,y_train,X_test,y_test, minority, majority):
    D = X_train
    # D.insert(len(X_train.columns), y_train.name, y_train)
    Do = D.iloc[np.where(D[D.columns[-1]] == minority)]
    Dj = D.iloc[np.where(D[D.columns[-1]] == majority)]
    sil = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(Do)
        labels = kmeans.labels_
        sil.append(silhouette_score(Do, labels, metric='euclidean'))
    k = sil.index(max(sil)) + 2
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Do)
    ratio = int(len(Dj) / len(Do))
    D_total = pd.DataFrame(columns=list(D.columns))
    for j in range(0, k):
        cluster = Do[kmeans.labels_ == j]
        cluster_X = cluster[cluster.columns[:-1]]
        C = np.mean(cluster_X)
        dist = {}
        for r in range(len(cluster)):
            dist[r] = np.linalg.norm(cluster_X.iloc[r, :] - C)
        r_min = min(dist, key=dist.get)
        Dmin = cluster_X.iloc[r_min, :]
        n = (ratio - 1) * len(cluster_X)
        ran = []
        for i in range(n):
            ran.append(random.random())
        data_new = pd.DataFrame(columns=list(D.columns))
        for i in ran:
            temp = Dmin + i * C
            data_new = data_new.append(temp, ignore_index=True)
        last_column_value = [minority] * n
        last_column_name = list(D.columns)[-1]
        data_new[last_column_name] = last_column_value
        cluster_new = cluster.append(data_new, ignore_index=True)
        D_total = D_total.append(cluster_new, ignore_index=True)
    D_total = D_total.append(Dj, ignore_index=True)
    y_train = D_total[D_total.columns[-1]]
    X_train = D_total[D_total.columns[:-1]]
    algorithmApply(X_train, y_train, X_test, y_test)

def algorithmApply(X_train,y_train,X_test,y_test):
    LogisticRegession(X_train, y_train, X_test, y_test)
    # RandomForestClassifier(X_train, y_train, X_test, y_test)
    # NaiveBayes(X_train, y_train, X_test, y_test)
    # DecisionTree(X_train, y_train, X_test, y_test)
    # SVM(X_train, y_train, X_test, y_test)
    # KNN(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # data = 'data\imbalanced_data\oil.csv'
    data = 'data\imbalanced_data\wine_quality.csv'
    # data = 'data\imbalanced_data\pima-indians-diabetes.csv'
    # data = 'data\imbalanced_data\letter_img.csv'
    minority = 1
    majority = -1
    # X_train,y_train,X_test,y_test=loadDatasetsCreditCard()
    X_train,y_train,X_test,y_test=loadDatasets(data)
    print('Algorithm without overshampling:')
    algorithmApply(X_train,y_train,X_test,y_test)
    print('SMOTE:')
    algorithmApplySMOTE(X_train,y_train,X_test,y_test)
    # print('logisticRegessionNearMiss:')
    # algorithmApplyNearMiss(X_train,y_train,X_test,y_test)
    print('KmeansSMOTE:')
    algorithmApplyKmeansSMOTE(X_train,y_train,X_test,y_test)
    print('CIR:')
    CIR(X_train,y_train,X_test,y_test, minority, majority)
    print('KCIR:')
    KmeansCIR(X_train,y_train,X_test,y_test, minority, majority)