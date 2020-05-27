from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from numpy import arange
import numpy
from openpyxl import load_workbook
from sklearn.feature_selection import RFECV
from collections import namedtuple
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


def xgbtree(X_train, Y_train, seed=27):
    """computes the feature importance"""
    XGBOOST = xgb.XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                                subsample=0.8,
                                colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                seed=seed)

    clf = XGBOOST
    model = clf.fit(X_train, Y_train)
    important_features = model.get_booster().get_score(importance_type='gain')

    return important_features, model


# The support vectors machine classifier
def svc_classification(X_train, Y_train, X_test, state=20):
    """A function that applies grid and random search to tune model and also gives a prediction"""
    # Creating a score and parameters to search from

    scoring = {"f1": "f1"}

    grid_param2 = [{"kernel": ["linear"], "C": arange(0.01, 1.1, 0.3)}, {"kernel": ["linear"], "C": range(1, 10, 2)},
                   {"kernel": ["rbf"], "C": arange(0.01, 1.1, 0.3), "gamma": arange(0.01, 1.1, 0.3)},
                   {"kernel": ["rbf"], "C": arange(1, 10, 2), "gamma": arange(1, 10, 2)},
                   {"kernel": ["rbf"], "C": arange(1, 10, 2), "gamma": arange(0.01, 1.1, 0.3)},
                   {"kernel": ["rbf"], "C": arange(0.01, 1.1, 0.3), "gamma": range(1, 10, 2)}]

    random_param = [{"kernel": ["linear"], "C": uniform(1, 10)}, {"kernel": ["linear"], "C": uniform(0.01, 0.99)},
                    {"kernel": ["rbf"], "C": uniform(1, 10), "gamma": uniform(1, 10)},
                    {"kernel": ["rbf"], "C": uniform(0.01, 0.99), "gamma": uniform(0.01, 0.99)},
                    {"kernel": ["rbf"], "C": uniform(0.01, 0.99), "gamma": uniform(1, 10)},
                    {"kernel": ["rbf"], "C": uniform(0.01, 0.99), "gamma": uniform(1, 10)}]

    # Model setting

    svc_grid = GridSearchCV(SVC(class_weight="balanced"), grid_param2, scoring=scoring, refit="f1", cv=10)
    svc_random = RandomizedSearchCV(SVC(class_weight="balanced"), random_param, scoring=scoring, refit="f1", cv=10,
                                    random_state=state)

    # Model training
    fitted_grid = svc_grid.fit(X_train, Y_train)
    fitted_random = svc_random.fit(X_train, Y_train)

    # Model predictions
    y_random = fitted_random.best_estimator_.predict(X_test)
    y_grid = fitted_grid.best_estimator_.predict(X_test)

    grid_train_predicted = fitted_grid.best_estimator_.predict(X_train)
    random_train_predicted = fitted_random.best_estimator_.predict(X_train)

    return fitted_grid, fitted_random, y_grid, y_random, random_train_predicted, grid_train_predicted


def print_score(fitted_grid, fitted_random, Y_test, y_random, y_grid, random_train_predicted, grid_train_predicted,
                Y_train, name=None):
    """ The function prints the scores of the models and the prediction performance """

    # Model comparison
    grid_score = fitted_grid.best_score_
    grid_params = fitted_grid.best_params_
    random_score = fitted_random.best_score_
    random_params = fitted_random.best_params_

    # print(f"best grid score {name}:{grid_score}, best grid parameters {name}: {grid_params}")
    # print(f"best random scores {name}: {random_score}, best random parameters {name}: {random_params}")

    # Training scores
    random_train_confusion = confusion_matrix(Y_train, random_train_predicted)
    grid_train_confusion = confusion_matrix(Y_train, grid_train_predicted)
    # print(f"grid training matrix {name}:{grid_train_confusion}")
    # print(f"random training matrix {name}: {random_train_confusion}")

    # Test metrics
    random_confusion = confusion_matrix(Y_test, y_random)
    grid_confusion = confusion_matrix(Y_test, y_grid)
    random_matthews = matthews_corrcoef(Y_test, y_random)
    grid_matthews = matthews_corrcoef(Y_test, y_grid)
    random_accuracy = accuracy_score(Y_test, y_random)
    grid_accuracy = accuracy_score(Y_test, y_grid)
    random_f1 = f1_score(Y_test, y_random, average="weighted")
    grid_f1 = f1_score(Y_test, y_grid, average="weighted")
    random_precision = precision_score(Y_test, y_random, average="weighted")
    grid_precision = precision_score(Y_test, y_grid, average="weighted")
    random_recall = recall_score(Y_test, y_random, average="weighted")
    grid_recall = recall_score(Y_test, y_grid, average="weighted")

    """print(f"random confusion matrix {name}: {random_confusion}, grid confusion matrix {name}: {grid_confusion}")
    print(f"random Mathew coefficient {name}: {random_matthews}, grid mathiews coeficient {name}: {grid_matthews}")
    print(f"random accuracy {name}: {random_accuracy}, grid accuracy {name}: {grid_accuracy}")
    print(f"random f1 score {name}: {random_f1}, grid f1 score {name}: {grid_f1}")
    print(f"random precision score {name}: {random_precision}, grid precision score {name}: {grid_precision}")
    print(f"random recall score {name}: {random_recall}, grid recall score {name}: {grid_recall}")"""

    return grid_score, grid_params, grid_confusion, grid_accuracy, grid_f1, grid_precision, grid_recall, \
           grid_matthews, random_score, random_params, random_confusion, random_accuracy, random_matthews, \
           random_f1, random_precision, random_recall, random_train_confusion, grid_train_confusion


def nested_cv(X, Y, name):
    """Performs something similar to a nested cross-validation"""

    metric_list = []
    model_list = []
    parameter_list = []
    random_state = [30, 42, 50, 70, 85, 90, 100, 200, 300, 400]

    for states in random_state:
        name += 1
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=states, stratify=Y)
        fitted_grid, fitted_random, y_grid, y_random, random_train_predicted, grid_train_predicted = svc_classification(
            X_train, Y_train, X_test, states)

        grid_score, grid_params, grid_confusion, grid_accuracy, grid_f1, grid_precision, grid_recall, grid_matthews, \
        random_score, random_params, random_confusion, random_accuracy, random_matthews, random_f1, random_precision, \
        random_recall, random_train_confusion, grid_train_confusion = print_score(
            fitted_grid, fitted_random, Y_test, y_random, y_grid, random_train_predicted, grid_train_predicted, Y_train,
            name)

        model_list.append([fitted_grid, fitted_random, y_grid, y_random])

        metric_list.append(
            [grid_score, grid_accuracy, grid_f1, grid_precision,
             grid_recall, grid_matthews, random_score,
             random_accuracy, random_matthews, random_f1, random_precision, random_recall])

        parameter_list.append([grid_params, grid_confusion, random_params, random_confusion,
                               random_train_confusion, grid_train_confusion])

    return model_list, metric_list, parameter_list, random_state


def mean_nested(X, Y, name):
    """From the results of the nested_CV it computes the means of the different performance metrics"""
    model_list, metric_list, parameter_list, random_state = nested_cv(X, Y, name)
    score_record = namedtuple("scores", ["grid_score", "grid_accuracy", "grid_f1", "grid_precision", "grid_recall",
                                         "grid_matthews", "random_score", "random_accuracy", "random_matthews",
                                         "random_f1", "random_precision", "random_recall"])

    parameter_record = namedtuple("parameters", ["grid_params", "grid_confusion", "random_params", "random_confusion",
                                                 "random_train_confusion", "grid_train_confusion"])

    model_record = namedtuple("models", ["fitted_grid", "fitted_random", "y_grid", "y_random"])

    array = numpy.array(metric_list)
    mean = numpy.mean(array, axis=0)

    named_parameters = [parameter_record(*z) for z in parameter_list]
    named_mean = score_record(*mean)
    named_records = [score_record(*y) for y in metric_list]
    named_models = [model_record(*d) for d in model_list]

    return named_mean, named_models, named_parameters, named_records, random_state


def unlisting(named_parameters, named_records):
    """ A function that separates all the scores in independent lists"""
    # Getting all scores random search
    r_mathew = [x.random_matthews for x in named_records]
    r_accuracy = [x.random_accuracy for x in named_records]
    r_f1 = [x.random_f1 for x in named_records]
    r_precision = [x.random_precision for x in named_records]
    r_recall = [x.random_recall for x in named_records]
    r_cv_score = [x.random_score for x in named_records]

    # Getting all scores grid search
    g_mathew = [x.grid_matthews for x in named_records]
    g_accuracy = [x.grid_accuracy for x in named_records]
    g_f1 = [x.grid_f1 for x in named_records]
    g_precision = [x.grid_precision for x in named_records]
    g_recall = [x.grid_recall for x in named_records]
    g_cv_score = [x.grid_score for x in named_records]

    # Hyperparameters grid search
    g_kernel = [y.grid_params["kernel"] for y in named_parameters]
    g_C = [y.grid_params["C"] for y in named_parameters]
    g_gamma = [y.grid_params["gamma"] if y.grid_params.get("gamma") else 0 for y in named_parameters]

    # Hyperparameters random search
    r_kernel = [y.random_params["kernel"] for y in named_parameters]
    r_C = [y.random_params["C"] for y in named_parameters]
    r_gamma = [y.random_params["gamma"] if y.random_params.get("gamma") else 0 for y in named_parameters]

    return r_mathew, r_accuracy, r_f1, r_precision, r_recall, g_mathew, g_accuracy, g_f1, g_precision, g_recall, \
           g_kernel, g_C, g_gamma, r_kernel, r_C, r_gamma, r_cv_score, g_cv_score


def to_dataframe(named_parameters, named_records, random_state):
    matrix = namedtuple("confusion_matrix", ["true_n", "false_p", "false_n", "true_p"])

    r_mathew, r_accuracy, r_f1, r_precision, r_recall, g_mathew, g_accuracy, g_f1, g_precision, g_recall, \
    g_kernel, g_C, g_gamma, r_kernel, r_C, r_gamma, r_cv_score, g_cv_score = unlisting(named_parameters, named_records)

    # Taking the confusion matrix
    g_test_confusion = [matrix(*x.grid_confusion.ravel()) for x in named_parameters]
    r_test_confusion = [matrix(*x.random_confusion.ravel()) for x in named_parameters]

    g_training_confusion = [matrix(*x.grid_train_confusion.ravel()) for x in named_parameters]
    r_training_confusion = [matrix(*x.random_train_confusion.ravel()) for x in named_parameters]

    # Separating confusion matrix into individual elements
    g_test_true_n = [y.true_n for y in g_test_confusion]
    g_test_false_p = [y.false_p for y in g_test_confusion]
    g_test_false_n = [y.false_n for y in g_test_confusion]
    g_test_true_p = [y.true_p for y in g_test_confusion]

    g_training_true_n = [z.true_n for z in g_training_confusion]
    g_training_false_p = [z.false_p for z in g_training_confusion]
    g_training_false_n = [z.false_n for z in g_training_confusion]
    g_training_true_p = [z.true_p for z in g_training_confusion]

    r_test_true_n = [y.true_n for y in r_test_confusion]
    r_test_false_p = [y.false_p for y in r_test_confusion]
    r_test_false_n = [y.false_n for y in r_test_confusion]
    r_test_true_p = [y.true_p for y in r_test_confusion]

    r_training_true_n = [z.true_n for z in r_training_confusion]
    r_training_false_p = [z.false_p for z in r_training_confusion]
    r_training_false_n = [z.false_n for z in g_training_confusion]
    r_training_true_p = [z.true_p for z in g_training_confusion]

    r_dataframe = pd.DataFrame([random_state, r_kernel, r_C, r_gamma, r_test_true_n, r_test_true_p,
                                r_test_false_p, r_test_false_n, r_training_true_n, r_training_true_p,
                                r_training_false_p,
                                r_training_false_n, r_mathew, r_f1, r_cv_score])

    g_dataframe = pd.DataFrame([random_state, g_kernel, g_C, g_gamma, g_test_true_n, g_test_true_p,
                                g_test_false_p, g_test_false_n, g_training_true_n, g_training_true_p,
                                g_training_false_p,
                                g_training_false_n, g_mathew, g_f1, g_cv_score])

    r_dataframe = r_dataframe.transpose()
    r_dataframe.columns = ["random_state", "kernel", "C", "gamma",
                           "test_tn", "test_tp", "test_fp", "test_fn", "train_tn", "train_tp", "train_fp",
                           "train_fn", "Mathews", "F1", "CV_F1"]

    g_dataframe = g_dataframe.transpose()
    g_dataframe.columns = ["random_state", "kernel", "C", "gamma", "test_tn",
                           "test_tp", "test_fp", "test_fn", "train_tn", "train_tp", "train_fp",
                           "train_fn", "Mathews", "F1", "CV_F1"]
    return r_dataframe.set_index("random_state"), g_dataframe.set_index("random_state")


def view_setting():
    """ Sets the console view of how many columns the console displays"""
    desired_width = 320
    pd.set_option('display.width', desired_width)
    numpy.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 14)


def writing(dataset, sheet_name):
    book = load_workbook('score_results.xlsx')
    writer = pd.ExcelWriter('score_results.xlsx', engine='openpyxl')
    writer.book = book

    ## ExcelWriter for some reason uses writer.sheets to access the sheet.
    ## If you leave it empty it will not know that sheet Main is already there
    ## and will create a new sheet.
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    dataset.to_excel(writer, sheet_name=f'{sheet_name}')
    writer.save()
    writer.close()


# Loading the excel files
global_score = pd.read_excel("sequences.xlsx", index_col=0, sheet_name="global")
local_score = pd.read_excel("sequences.xlsx", index_col=0, sheet_name="local")

# generating X and Y
Y = global_score["label"].copy()
X = global_score.drop(["seq", "label"], axis=1)

standard_X = StandardScaler().fit_transform(X)
feature_names = [i for i in X.columns]

# Trying the nested CV
s_named_mean, s_model_list, s_named_parameters, s_named_records, s_random_state = mean_nested(standard_X, Y, 0)

# Generates the dataframe
r_dataframe, g_dataframe = to_dataframe(s_named_parameters, s_named_records, s_random_state)

view_setting()
