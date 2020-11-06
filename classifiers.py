import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

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
        print(y_pred)
        # Calculate confusion matrix and model performance
        cm = confusion_matrix(y_test, y_pred)
        acc_i = (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    
        print('acc = ', acc_i)

        acc += acc_i 

    acc = acc/5
    print('ACC = ', acc)

def sl(x,y):
    clf = svm.SVC(kernel = 'linear')
    clf.fit(x,y)

    return clf

def svd_lineal(x,y):
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle = True)
    clf = svm.SVC(kernel = 'linear')

    train(x,y,clf,kf)

    return clf

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

def nn_singlelayer_3c(x,y):
    n_features = x.shape[1]
    kf = KFold(n_splits=5, shuffle = True)
    acc = 0
    recall = np.array([0., 0., 0.])

    y = y-101

    for train_index, test_index in kf.split(x):
        # Training phase
        x_train = x[train_index, :]
        #y_train = y[train_index]
        yC_train = y[train_index]
        yC_train = np_utils.to_categorical(yC_train)

        clf = Sequential()
        clf.add(Dense(8, input_dim=n_features, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(3, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer='adam')
        clf.fit(x_train, yC_train, epochs=100, batch_size=8, verbose=0)    

        # Test phase
        x_test = x[test_index, :]
        #y_test = y[test_index]
        yC_test = y[test_index]
        y_pred = np.argmax(clf.predict(x_test), axis=-1)
        
        cm = confusion_matrix(yC_test, y_pred)
        #print(cm)
        acc += (cm[0,0]+cm[1,1]+cm[2,2])/len(yC_test)
        recall[0] += cm[0,0]/(cm[0,0] + cm[0,1] + cm[0,2])
        recall[1] += cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
        recall[2] += cm[2,2]/(cm[2,0] + cm[2,1] + cm[2,2])
    acc = acc/5
    print(recall)
    recall = recall/5
    print('Acc: ', acc)
    print('Recall:', recall)

def nn_multilayer_3c(x,y):
    n_features = x.shape[1]
    kf = KFold(n_splits=5, shuffle = True)
    acc = 0
    recall = np.array([0., 0., 0.])

    y = y-101

    for train_index, test_index in kf.split(x):
        # Training phase
        x_train = x[train_index, :]
        #y_train = y[train_index]
        yC_train = y[train_index]
        yC_train = np_utils.to_categorical(yC_train)

        clf = Sequential()
        clf.add(Dense(8, input_dim=n_features, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(3, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer='adam')
        clf.fit(x_train, yC_train, epochs=100, batch_size=8, verbose=0)    

        # Test phase
        x_test = x[test_index, :]
        #y_test = y[test_index]
        yC_test = y[test_index]
        y_pred = np.argmax(clf.predict(x_test), axis=-1)
        
        cm = confusion_matrix(yC_test, y_pred)
        #print(cm)
        acc += (cm[0,0]+cm[1,1]+cm[2,2])/len(yC_test)
        recall[0] += cm[0,0]/(cm[0,0] + cm[0,1] + cm[0,2])
        recall[1] += cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
        recall[2] += cm[2,2]/(cm[2,0] + cm[2,1] + cm[2,2])
    acc = acc/5
    print(recall)
    recall = recall/5
    print('Acc: ', acc)
    print('Recall:', recall)

def nn_singlelayer_2c(x,y):
    n_features = x.shape[1]
    kf = KFold(n_splits=5, shuffle = True)
    acc = 0
    recall = np.array([0., 0.])

    y = y-101

    for train_index, test_index in kf.split(x):
        # Training phase
        x_train = x[train_index, :]
        #y_train = y[train_index]
        yC_train = y[train_index]        

        clf = Sequential()
        clf.add(Dense(8, input_dim=n_features, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(1, activation='sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        clf.fit(x_train, yC_train, epochs=100, batch_size=8, verbose=0)    

        # Test phase
        x_test = x[test_index, :]
        #y_test = y[test_index]
        yC_test = y[test_index]
        y_pred = (clf.predict(x_test)>0.5).astype("int32")
        
        cm = confusion_matrix(yC_test, y_pred)
        #print(cm)
        acc += (cm[0,0]+cm[1,1])/len(yC_test)
        recall[0] += cm[0,0]/(cm[0,0] + cm[0,1])
        recall[1] += cm[1,1]/(cm[1,0] + cm[1,1])
    acc = acc/5
    print(recall)
    recall = recall/5
    print('Acc: ', acc)
    print('Recall:', recall)

def nn_multilayer_2c(x,y):
    n_features = x.shape[1]
    kf = KFold(n_splits=5, shuffle = True)
    acc = 0
    recall = np.array([0., 0.])

    y = y-101

    for train_index, test_index in kf.split(x):
        # Training phase
        x_train = x[train_index, :]
        #y_train = y[train_index]
        yC_train = y[train_index]        

        clf = Sequential()
        clf.add(Dense(8, input_dim=n_features, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(8, activation='relu'))
        clf.add(Dense(1, activation='sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        clf.fit(x_train, yC_train, epochs=100, batch_size=8, verbose=0)    

        # Test phase
        x_test = x[test_index, :]
        #y_test = y[test_index]
        yC_test = y[test_index]
        y_pred = (clf.predict(x_test)>0.5).astype("int32")
        
        cm = confusion_matrix(yC_test, y_pred)
        #print(cm)
        acc += (cm[0,0]+cm[1,1])/len(yC_test)
        recall[0] += cm[0,0]/(cm[0,0] + cm[0,1])
        recall[1] += cm[1,1]/(cm[1,0] + cm[1,1])
    acc = acc/5
    print(recall)
    recall = recall/5
    print('Acc: ', acc)
    print('Recall:', recall)

