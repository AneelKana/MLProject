import csv
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from memory_profiler import memory_usage
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def loadProcessData():
    data = []
    with open('wdbc.data', mode ='r') as file:
        csv_data = csv.reader(file)
        for line in csv_data:
            data.append(line)
    data = np.array(data)

    X = data[:, 2:]
    y = (data[:, 1] == 'M').astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=0)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def setUpRun(X_train, y_train):
    estimator = DecisionTreeClassifier(random_state = 0)
    search_space = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }
    GS = GridSearchCV(
        estimator = estimator,
        param_grid = search_space,
        scoring = 'accuracy',
        refit = 'accuracy',
        cv = 5,
        verbose = 0, #0-3
    )

    mem_usage = memory_usage((GS.fit, (X_train, y_train)))
    return GS, mem_usage

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadProcessData()
    GS, mem_usage = setUpRun(X_train, y_train)

    print('Best Estimator:', GS.best_estimator_)
    print('Best parameters:', GS.best_params_)
    print('Best estimator\'s validation accuracy:', np.round(GS.best_score_, 4))
    print('Best estimator\'s test accuracy:', np.round(np.mean(GS.best_estimator_.predict(X_test) == y_test), 4))
    print(
        'Training time: ',
        np.round(np.mean(GS.cv_results_['mean_fit_time']), 3),
        'sec ±',
        np.round(np.mean(GS.cv_results_['std_fit_time']), 3),
        '\nInference time:',
        np.round(np.mean(GS.cv_results_['mean_score_time']), 3),
        'sec ±',
        np.round(np.mean(GS.cv_results_['std_score_time']), 3),
        '\nMemory usage:  ',
        np.round(np.mean(mem_usage), 1),
        'MiB ±',
        np.round(np.std(mem_usage), 1),
    )