import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from memory_profiler import memory_usage

def loadProcessData():
    data = []
    with open('wdbc.data', mode ='r') as file:
        csv_data = csv.reader(file)
        for line in csv_data:
            data.append(line)
    data = np.array(data)

    X = data[:, 2:].astype(np.float64)
    y = (data[:, 1] == 'M').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def setUpRun(X_train, y_train):
    estimator = MLPClassifier(random_state=0, max_iter=10000)
    search_space = {
        'hidden_layer_sizes': [
            (3,), (10,), (30,), (100,),
            (3, 3), (10, 10), (30, 30), (100, 100),
            (3, 3, 3), (10, 10, 10), (30, 30, 30), (100, 100, 100),
            (3, 3, 3, 3), (10, 10, 10, 10), (30, 30, 30, 30), (100, 100, 100, 100),
        ],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.01, 0.003, 0.001, 0.0003]
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