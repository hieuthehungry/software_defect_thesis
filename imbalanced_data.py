import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, balanced_accuracy_score
from utils import read_arff, convert_label
import os

# Những module nào cần import thì nên import hết ở đầu file
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from kmeans_smote import KMeansSMOTE

def printDatasetInfo(X_train,y_train,X_test,y_test):
    print("Number transactions X_train dataset: ", X_train.shape)
    print("Number transactions y_train dataset: ", y_train.shape)
    print("Number transactions X_test dataset: ", X_test.shape)
    print("Number transactions y_test dataset: ", y_test.shape)

def printDatasetInfoAfter(X_train,y_train):
    print('After OverSampling, the shape of train_X: {}'.format(X_train.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train.shape))
    print("After OverSampling, counts of label 'True': {}".format(sum(y_train == True)))
    print("After OverSampling, counts of label 'False': {}".format(sum(y_train == False)))
# -------------------------------------

def loadDatasets(path, file_type = "csv"):
    if file_type == "csv":
        data = pd.read_csv(path)    
    elif file_type == "arff":
        data = read_arff(path)
    print(len(data))
    y = data[data.columns[-1]].apply(convert_label)
    X = data[data.columns[:-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=0)
    return X_train,y_train,X_test,y_test
# ---------------------------------
# Tạo một hàm có tên là `evaluate_and_visualize` ở đây để đánh giá mô hình
# Cái khối lệnh này em thấy anh Vĩnh dùng đi dùng lại thì tốt nhất là tách nó ra thành hàm riêng, về sau dùng chỉ việc gọi thôi
def evaluate_and_visualize(model, y_test,predictions):    
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test, predictions))
    print("AUC Score: ",roc_auc_score(y_test,predictions))
    print("balanced_accuracy_score: ",balanced_accuracy_score(y_test,predictions))
    clf_disp = plot_roc_curve(model, X_test, y_test)
    clf_disp.figure_.suptitle("ROC curve comparison")    
    plt.show()

# Hàm này để huấn luyện và đánh giá co tất cả mô hình
def training_and_evaluate(model, X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    predictions=model.predict(X_test)
    evaluate_and_visualize(model, y_test,predictions)

# Các phương pháp balance data sẽ được tổng hợp trong hàm này
def balance_dataset(X_train,y_train, balancer = "smote"):
    assert balancer in ["smote", "adasyn", "near_miss", "kmeans_smote"]

    if balancer == "smote":
        bal=SMOTE(random_state=0)
    elif balancer == "adasyn":
        bal = ADASYN(random_state=42,n_neighbors=50)
    elif balancer == "near_miss":
        bal = NearMiss()
    elif balancer == "kmeans_smote":
        kmeans_smote = KMeansSMOTE(kmeans_args={'n_clusters': 100}, smote_args={'k_neighbors': 13})

    X_train_res,y_train_res = bal.fit_sample(X_train,y_train)
    
    return X_train_res, y_train_res


if __name__ == "__main__":
    lr = LogisticRegression()
    tree = DecisionTreeClassifier()
    svm = SVC()
    knn = KNeighborsClassifier(n_neighbors=11)
    random_forest = RandomForestClassifier(n_estimators=410, class_weight="balanced")
    gboost = GradientBoostingClassifier(n_estimators = 300,learning_rate=1.0,max_depth=1, random_state=0)
    gaussian_nb = GaussianNB()
    

    # Các phương pháp ensemble learning đã tích hợp sẵn phần cân bằng dữ liệu ta để đây
    brf = BalancedRandomForestClassifier(n_estimators=500,max_depth=17, random_state=3)
    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),sampling_strategy='auto',replacement=False,random_state=0)
    eec = EasyEnsembleClassifier(n_estimators=200)
    rus_boost = RUSBoostClassifier(n_estimators=15)
    
    # Stacking define hơi mất công chút
    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svc', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))]
    stack = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(random_state=42))

    data = 'data/thesis_data/arff/pc1.arff.txt'
    X_train,y_train,X_test,y_test=loadDatasets(data, file_type ="arff")

    import numpy as np
    print(np.unique(y_train))

    printDatasetInfo(X_train,y_train,X_test,y_test)
    X_train_balance, Y_train_balance = balance_dataset(X_train, y_train, balancer = "adasyn")
    printDatasetInfoAfter(X_train,y_train)

    # Mình muốn thử nghiệm stacking thì mình truyền nó vào vào hàm dưới đây
    training_and_evaluate(stack, X_train_balance, Y_train_balance, X_test, y_test)
    