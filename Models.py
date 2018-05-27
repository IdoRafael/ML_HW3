import os
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from DataPreparation import split_label
from ReadWrite import read_data


MODELS_FOLDER = 'Models'


def load_prepared_data():
    return (read_data(x) for x in ['train.csv', 'validate.csv', 'test.csv'])


def optimize_models_parameters(train_x, train_y):
    score = make_scorer(f1_score, average='weighted')
    cv = 5

    """
    # SVM
    print('SVC')
    start_time = time.time()
    svc = GridSearchCV(
        SVC(),
        [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3], 'C': [1e3, 1e4, 1e5, 1e6, 1e7]},
         {'kernel': ['linear'], 'C': [1e1, 1e2, 1e3]}],
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)

    # KNN
    print('KNN')
    start_time = time.time()
    knn = GridSearchCV(
        KNeighborsClassifier(),
        [{'n_neighbors': [1, 3, 5, 7]}],
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)

    # DecisionTreeClassifier
    print('DECISION_TREE')
    start_time = time.time()
    tree = GridSearchCV(
        DecisionTreeClassifier(),
        [{'max_depth': [None, 7, 8, 9, 10, 11, 12, 13]}],
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)
    """

    """'max_features': []"""

    # DecisionTreeClassifier
    print('RANDOM_FOREST')
    start_time = time.time()
    random_forest = GridSearchCV(
        RandomForestClassifier(),
        [{'max_depth': [13, 14, 15, 16, 17, 18, 19, None],
          'max_features': [None, 'sqrt', 'log2', 5, 6, 7, 8, 9, 10]}],
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)

    # return [svc, knn, tree, random_forest], ['SVC', 'KNN', 'DECISION_TREE', 'RANDOM_FOREST']
    return [random_forest], ['RANDOM_FOREST']


def get_best_model(train, validate, models):
    return 12


def predict_test_and_save_results(trained_model, test):
    return None


def save_models(models, names):
    for model, name in zip(models, names):
        save_model(model, name)


def save_model(model, name):
    # Store data (serialize)
    with open(os.path.join(MODELS_FOLDER, name + '.pickle'), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_model(name):
    # Load data (deserialize)
    with open(os.path.join(MODELS_FOLDER, name + '.pickle'), 'rb') as handle:
        return pickle.load(handle)


def print_model(model, name):
    print('=' * 100)
    print(name)
    print("Best parameters set found on development set:")
    print(model.best_params_)
    print()
    print("Grid scores on development set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('=' * 100)


def load_optimize_fit_select_and_predict():
    train, validate, test = load_prepared_data()

    train_x, train_y = split_label(train)
    models, names = optimize_models_parameters(train_x, train_y)

    save_models(models, names)

    best_model_trained = get_best_model(train, validate, models)

    predict_test_and_save_results(best_model_trained, test)

