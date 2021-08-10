import numpy as np
import pandas as pd
import random
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold

def loadDatasets(data):
    data = pd.read_csv(data)
    y = data[data.columns[-1]]
    X = data[data.columns[:-1]]
    return X, y

def ApplyAlgorithmn(D, X, y, n, irt, kfold, minority, majority, model, type):
    # Innit data
    accuracy = 0
    precision = 0
    recall = 0
    f1score = 0

    # KFold
    kf = KFold(n_splits=kfold, random_state=42)
    for train_index, test_index in kf.split(X):
        D_train = D.iloc[train_index, :]
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if type == 'SMOTE':
            from imblearn.over_sampling import SMOTE
            sm = SMOTE()
            X_train, y_train = sm.fit_resample(X_train, y_train)
        elif type == 'KmeansSMOTE':
            from imblearn.over_sampling import KMeansSMOTE
            kmeans_smote = KMeansSMOTE()
            X_train, y_train = kmeans_smote.fit_resample(X_train, y_train)
        elif type == 'CIR':
            X_train, y_train = CIR(D_train, minority, majority)
        elif type == 'Kmeans_CIR':
            kmeans, k = ApplyKmean(D_train)
            X_train, y_train = KmeansCIR(D_train, n, irt, minority, majority, kmeans, k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy += accuracy_score(y_test, predictions)
        precision += precision_score(y_test, predictions, average='weighted')
        recall += recall_score(y_test, predictions, average='weighted')
        f1score += f1_score(y_test, predictions, average='weighted')
    accuracy /= kfold
    precision /= kfold
    recall /= kfold
    f1score /= kfold
    print('accuracy: {0}, precision: {1}, recall: {2}, f1_score: {3}'.format(accuracy,
                                                                             precision,
                                                                             recall,
                                                                             f1score))

def CIR(D_train, minority, majority):
    Do = D_train.iloc[np.where(D_train[D_train.columns[-1]] == minority)]
    Dj = D_train.iloc[np.where(D_train[D_train.columns[-1]] == majority)]
    Do_X = Do[Do.columns[:-1]]
    C = np.mean(Do_X)
    dist = {}
    for r in range(len(Do)):
        dist[r] = np.linalg.norm(Do_X.iloc[r, :] - C)
    r_min = min(dist, key=dist.get)
    Dmin = Do_X.iloc[r_min, :]
    n = len(Dj) - len(Do)
    k = []
    for i in range(n):
        k.append(random.random())
    data_new = []
    for i in k:
        temp = (Dmin + i * C).values.tolist()
        temp.append(minority)
        data_new.append(temp)
    Do_new = Do.append(pd.DataFrame(data_new, columns=D_train.columns), ignore_index=True)
    D_total = Dj.append(Do_new, ignore_index=True)
    y_train = D_total[D_total.columns[-1]]
    X_train = D_total[D_total.columns[:-1]]
    return X_train, y_train

def KmeansCIR(D_train, n, irt, minority, majority, kmeans, k):
    D_total = pd.DataFrame(columns=list(D_train.columns))
    density_dict = {}
    density_sum = 0
    # Loop for each cluster
    for j in range(0, k):
        D_cluster = D_train[kmeans.labels_ == j]
        # X_cluster = D_cluster[D_cluster.columns[:-1]]
        y_cluster = D_cluster[D_cluster.columns[-1]]

        # Step 1.2
        number_majority = y_cluster[y_cluster == majority].count()
        number_minority = y_cluster[y_cluster == minority].count()
        ratio = (number_majority + 1) / (number_minority + 1)

        # Step 1.3
        if ratio < irt and number_minority > 1:
            # Step 2.1
            D_cluster_minority = D_cluster.iloc[np.where(D_cluster[D_cluster.columns[-1]] == minority)]
            D_cluster_minority_X = D_cluster_minority[D_cluster_minority.columns[:-1]]
            distance_matrix = euclidean_distances(D_cluster_minority_X, D_cluster_minority_X)

            # Step 2.2
            distance_mean = np.sum(distance_matrix) / (len(distance_matrix) * 2)
            # print(distance_mean)

            # Step 2.3
            density = len(D_cluster_minority) / distance_mean

            # Step 2.4
            sparse = 1 / density

            # Step 2.5
            density_dict[j] = sparse
            density_sum += sparse
    for j in range(0, k):
        if j in density_dict.keys():
            # Step 2.6
            weight = density_dict[j] / density_sum

            # Step 3.1
            number_sample_increased = math.floor(n * weight)

            # Step 3.2
            D_cluster = D_train[kmeans.labels_ == j]
            D_cluster_minority = D_cluster.iloc[np.where(D_cluster[D_cluster.columns[-1]] == minority)]
            # D_cluster_majority = D_cluster.iloc[np.where(D_cluster[D_cluster.columns[-1]] == majority)]
            D_cluster_minority_X = D_cluster_minority[D_cluster_minority.columns[:-1]]

            # Step 3.3
            C = np.mean(D_cluster_minority_X)

            # Step 3.4
            dist = {}
            for r in range(len(D_cluster_minority)):
                dist[r] = np.linalg.norm(D_cluster_minority_X.iloc[r, :] - C)

            # Step 3.5
            r_min = min(dist, key=dist.get)
            Dmin = D_cluster_minority_X.iloc[r_min, :]

            # Step 3.6
            random_array = []
            for i in range(number_sample_increased):
                random_array.append(random.random())
            data_new = []
            for i in random_array:
                temp = (Dmin + i * C).values.tolist()
                temp.append(minority)
                data_new.append(temp)
            cluster_new = D_cluster.append(pd.DataFrame(data_new, columns=D_train.columns), ignore_index=True)
            D_total = D_total.append(cluster_new, ignore_index=True)
        else:
            D_total = D_total.append(D_train[kmeans.labels_ == j], ignore_index=True)
    y_train = D_total[D_total.columns[-1]]
    X_train = D_total[D_total.columns[:-1]]
    return X_train, y_train

def ApplyKmean(D_train):
    X_train = D_train[D_train.columns[:-1]]
    sil = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train)
        labels = kmeans.labels_
        sil.append(silhouette_score(X_train, labels, metric='euclidean'))
    k = sil.index(max(sil)) + 2
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_train)
    return kmeans, k

def SelectModel(algorithm):
    if algorithm == 'LogisticRegession':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif algorithm == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif algorithm == 'NaiveBayes':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif algorithm == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif algorithm == 'SVM':
        from sklearn.svm import SVC
        model = SVC()
    elif algorithm == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    return model

#file_name = '..\..\data\imbalanced_data\imbalanced-learn\_abalone.csv'
# file_name = '..\..\data\imbalanced_data\imbalanced-learn\ecoli.csv'
# file_name = '..\..\data\imbalanced_data\imbalanced-learn\letter_img.csv'
# file_name = '..\..\data\imbalanced_data\imbalanced-learn\oil.csv'
#file_name = '..\..\data\imbalanced_data\imbalanced-learn\optical_digits.csv'
# file_name = '..\..\data\imbalanced_data\imbalanced-learn\pen_digits.csv'
# file_name = '..\..\data\imbalanced_data\imbalanced-learn\satimage.csv'
#file_name = '..\..\data\imbalanced_data\imbalanced-learn\wine_quality.csv'
file_name = 'software_defect.csv'
print('file_name: {0}'.format(file_name))

data = pd.read_csv(file_name)
y = data[data.columns[-1]]
X = data[data.columns[:-1]]

minority = 1 # value of label minority
majority = -1 # value of label majority
kfold = 5 # number of kfold

listOfAlgorithm = ['LogisticRegession', 'RandomForest', 'NaiveBayes', 'DecisionTree', 'SVM', 'KNN']
listOfOverSample = ['Normal', 'SMOTE', 'KmeansSMOTE', 'CIR', 'Kmeans_CIR']
algorithmUse = listOfAlgorithm[1]
print('algorithmUse: {0}'.format(algorithmUse))

print('Normal: ')
ApplyAlgorithmn(data, X, y, 0, 0, kfold, minority, majority, SelectModel(algorithmUse), listOfOverSample[0])

print('SMOTE: ')
ApplyAlgorithmn(data, X, y, 0, 0, kfold, minority, majority, SelectModel(algorithmUse), listOfOverSample[1])

# print('KmeansSMOTE: ')
# ApplyAlgorithmn(file_name, list_n, list_irt, kfold, minority, majority, SelectModel(algorithmUse), listOfOverSample[2])

print('CIR: ')
ApplyAlgorithmn(data, X, y, 0, 0, kfold, minority, majority, SelectModel(algorithmUse), listOfOverSample[3])

print('Kmeans_CIR: ')
list_n = [5, 10, 20, 40, 80] # coefficient of the number of minority created
list_irt = [1, 2, 4, 8, 16] # coefficient of the number of minority created
ration = math.floor(y[y == majority].count() / y[y == minority].count())
for i in range(len(list_n)):
    for j in range(len(list_irt)):
        n =  list_n[i] * ration
        irt = list_irt[j] * ration
        print('n: {0}, irt: {1}'.format(n, irt))
        ApplyAlgorithmn(data, X, y, n, irt, kfold, minority, majority, SelectModel(algorithmUse), listOfOverSample[4])

