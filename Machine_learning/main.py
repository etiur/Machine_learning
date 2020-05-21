from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from numpy import arange
import numpy
from collections import namedtuple

mathew = make_scorer(matthews_corrcoef)
# The support vectors machine classifier
def svc_classification(X_train, Y_train, X_test):
    """A function that applies grid and random search to tune model and also gives a prediction"""
    # Creating a score and parameters to search from
    scoring = {"accuracy": "accuracy", "f1": "f1", "recall": "recall", "precision": "precision"}

    grid_param2 = [{"kernel": ["linear"], "C": arange(0.01, 1.1, 0.3)}, {"kernel": ["linear"], "C": range(1, 10, 2)},
                {"kernel": ["rbf"], "C": arange(0.01, 1.1, 0.3), "gamma": arange(0.01, 1.1, 0.3)},
               {"kernel": ["rbf"], "C": arange(1, 10, 2), "gamma": arange(1, 10, 2)},
               {"kernel": ["rbf"], "C": arange(1, 10, 2), "gamma": arange(0.01, 1.1, 0.3)},
               {"kernel": ["rbf"], "C": arange(0.01, 1.1, 0.3), "gamma": range(1, 10, 2)}]

    random_param = [{"kernel":["linear"], "C": uniform(1, 10)}, {"kernel":["linear"], "C": uniform(0.01, 0.99)},
                    {"kernel":["rbf"], "C": uniform(1, 10), "gamma": uniform(1, 10)},
                {"kernel":["rbf"], "C": uniform(0.01, 0.99), "gamma": uniform(0.01, 0.99)},
                    {"kernel":["rbf"], "C": uniform(0.01, 0.99), "gamma": uniform(1, 10)},
                    {"kernel":["rbf"], "C": uniform(0.01, 0.99), "gamma": uniform(1, 10)}]

    # Model setting

    svc_grid = GridSearchCV(SVC(), grid_param2, scoring=scoring, refit="f1", cv=10)
    svc_random = RandomizedSearchCV(SVC(), random_param, scoring=scoring, refit="f1", cv=10, random_state=20)

    #Model training
    fitted_grid = svc_grid.fit(X_train, Y_train)
    fitted_random = svc_random.fit(X_train, Y_train)

    #Model predictions
    y_random = fitted_random.predict(X_test)
    y_grid = fitted_grid.predict(X_test)

    return fitted_grid, fitted_random, y_grid, y_random

def print_score(fitted_grid, fitted_random, Y_test, y_random, y_grid):
    """ The function prints the scores of the models and the prediction performance """

    # Model comparison
    grid_score = fitted_grid.best_score_
    grid_params = fitted_grid.best_params_
    random_score = fitted_random.best_score_
    random_params = fitted_random.best_params_

    print(f"best grid score:{grid_score}, best grid parameters: {grid_params}")
    print(f"best random scores: {random_score}, best random parameters: {random_params}")

    # Metrics
    random_confusion = confusion_matrix(Y_test, y_random)
    grid_confusion = confusion_matrix(Y_test, y_grid)
    random_matthews = matthews_corrcoef(Y_test, y_random)
    grid_matthews = matthews_corrcoef(Y_test, y_grid)
    random_accuracy = accuracy_score(Y_test, y_random)
    grid_accuracy = accuracy_score(Y_test, y_grid)
    random_multiple = precision_recall_fscore_support(Y_test, y_random, average="weighted")
    grid_multiple = precision_recall_fscore_support(Y_test, y_grid, average= "weighted")
    print(f"best random confusion matrix: {random_confusion}, best grid confusion matrix: {grid_confusion}")
    print(f"best random Mathew coefficient: {random_matthews}, best grid mathiews coeficient: {grid_matthews}")
    print(f"best random accuracy: {random_accuracy}, best grid accuracy: {grid_accuracy}")
    print(f"best random multiple score: {random_multiple}, best grid multiple score: {grid_multiple}")

    return grid_score, grid_params, grid_confusion, grid_accuracy, grid_multiple, grid_matthews, random_score, random_params, random_confusion, random_accuracy, random_matthews, random_multiple

def nested_cv(X, Y):
    """Performs something similar to a nested cross-validation"""

    metric_list = []
    model_list = []
    for states in [30, 42, 50, 70, 80, 90, 100, 200, 300, 400]:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=states)
        fitted_grid, fitted_random, y_grid, y_random = svc_classification(X_train, Y_train, X_test)
        grid_score, grid_params, grid_confusion, grid_accuracy, grid_multiple, grid_matthews, random_score, random_params, random_confusion, random_accuracy, random_matthews, random_multiple = print_score(
            fitted_grid, fitted_random, Y_test, y_random, y_grid)

        model_list.append([fitted_grid, fitted_random, y_grid, y_random])
        metric_list.append(
            [grid_score, grid_params, grid_confusion, grid_accuracy, grid_multiple, grid_matthews, random_score,
             random_params, random_confusion, random_accuracy, random_matthews, random_multiple])

    return model_list, metric_list

def mean_nested(X, Y):
    """From the results of the nested_Cv it computes the means of the different performance metrics"""
    model_list, metric_list = nested_cv(X, Y)
    array = numpy.array(metric_list)
    mean = numpy.mean(array, axis=0)
    record = namedtuple("scores", ["grid_score", "grid_params", "grid_confusion", "grid_accuracy", "grid_multiple", "grid_matthews",
                                   "random_score", "random_params", "random_confusion", "random_accuracy", "random_matthews", "random_multiple"])
    named_mean = record(*mean)
    return named_mean, model_list

# Loading the excel files
global_score = pd.read_excel("sequences.xlsx", index_col=0, sheet_name="global")
local_score = pd.read_excel("sequences.xlsx", index_col=0, sheet_name="local")

# generating X and Y
Y = global_score["label"].copy()
X = global_score.drop(["seq", "label"], axis=1)

# Generate the model and the performance metrics
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
fitted_grid, fitted_random, y_grid, y_random = svc_classification(X_train, Y_train, X_test)
grid_score, grid_params, grid_confusion, grid_accuracy, grid_multiple, grid_matthews, random_score, random_params, random_confusion, random_accuracy, random_matthews, random_multiple = print_score(
            fitted_grid, fitted_random, Y_test, y_random, y_grid)
