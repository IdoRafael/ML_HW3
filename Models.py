import os
import pickle
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from DataPreparation import split_label
from ReadWrite import read_data

MODELS_FOLDER = 'Models'


def load_prepared_data():
    return (read_data(x) for x in ['train.csv', 'validate.csv', 'test.csv'])


def test_model(model, name, parameters, train_x, train_y):
    score = make_scorer(f1_score, average='weighted')
    cv = 5

    print(name)
    start_time = time.time()
    classifier = GridSearchCV(
        model,
        parameters,
        cv=cv, scoring=score,
        n_jobs=-1
    ).fit(train_x, train_y)
    print(time.time() - start_time)
    return classifier


def run_experiments(train_x, train_y, names):
    # SVM
    svc = test_model(
        SVC(),
        'SVC',
        [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3], 'C': [1e3, 1e4, 1e5, 1e6, 1e7]},
         {'kernel': ['linear'], 'C': [1e1, 1e2, 1e3]}],
        train_x, train_y
    )

    # KNN
    knn = test_model(
        KNeighborsClassifier(),
        'KNN',
        [{'n_neighbors': [1, 3, 5, 7]}],
        train_x, train_y
    )

    # DecisionTreeClassifier
    tree = test_model(
        DecisionTreeClassifier(),
        'DECISION_TREE',
        [{'max_depth': [None, 7, 8, 9, 10, 11, 12, 13]}],
        train_x, train_y
    )

    # RANDOM_FOREST
    random_forest = test_model(
        RandomForestClassifier(),
        'RANDOM_FOREST',
        [{'max_depth': [13, 14, 15, 16, 17, 18, 19, None],
          'max_features': [None, 'sqrt', 'log2', 5, 6, 7, 8, 9, 10]}],
        train_x, train_y
    )

    # GBC
    gbc = test_model(
        GradientBoostingClassifier(),
        'GBC',
        [{'max_depth': [3, 7, 13, None], 'max_features': [3, 7, 10, 13, None]}],
        train_x, train_y
    )

    # MLP
    mlp = test_model(
        MLPClassifier(),
        'MLP',
        [{'hidden_layer_sizes': [(15,), (15, 15,), (100, 100,), (500, 500,)],
          'alpha': [1e-4, 1.5e-4],
          'learning_rate_init': [1e-3, 1.5e-3]}],
        train_x, train_y
    )
    models = [svc, knn, tree, random_forest, gbc, mlp]
    save_models(models, names)

    return models


def load_experiments(names):
    models = [read_model(name) for name in names]
    for model, name in zip(models, names):
        print_model(model, name)
    return models


def optimize_models_parameters(train_x, train_y, rerun_experiments=False):
    names = ['SVC', 'KNN', 'DECISION_TREE', 'RANDOM_FOREST', 'GBC', 'MLP']
    models = run_experiments(train_x, train_y, names) if rerun_experiments else load_experiments(names)
    return models, names


def get_best_model(validate, models, names):
    return None


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

    # TODO - change to rerun_experiments=True when submitting HW
    models, names = optimize_models_parameters(train_x, train_y)

    best_model_trained = get_best_model(validate, models, names)

    predict_test_and_save_results(best_model_trained, test)
