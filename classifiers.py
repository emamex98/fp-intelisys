from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def train(x,y,clf,kf):
    acc = 0
    for train_index, test_index in kf.split(x):

        # Training phase
        x_train = x[train_index, :]
        y_train = y[train_index]
        clf.fit(x_train, y_train)

        # Test phase
        x_test = x[test_index, :]
        y_test = y[test_index]    
        y_pred = clf.predict(x_test)

        # Calculate confusion matrix and model performance
        cm = confusion_matrix(y_test, y_pred)
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
        print('acc = ', acc_i)

        acc += acc_i 

    acc = acc/5
    print('ACC = ', acc)

def svd_lineal(x,y):
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle = True)
    clf = svm.SVC(kernel = 'linear')

    train(x,y,clf,kf)

def svd_radial(x,y):
    kf = KFold(n_splits=5, shuffle = True)
    clf = svm.SVC(kernel = 'rbf')

    train(x,y,clf,kf)

def decision_tree(x,y):
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle = True)
    clf = DecisionTreeClassifier(random_state=0)

    train(x,y,clf,kf)

def knn(x,y):
    kf = KFold(n_splits=5, shuffle = True)
    clf = KNeighborsClassifier(n_neighbors=3)

    train(x,y,clf,kf)

