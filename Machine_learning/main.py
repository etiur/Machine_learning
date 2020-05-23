from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix, accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from numpy import arange
import numpy
from collections import namedtuple
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import pickle

mathew = make_scorer(matthews_corrcoef)

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

    svc_grid = GridSearchCV(SVC(class_weight="balanced"), grid_param2, scoring=scoring, refit="f1", cv=5)
    svc_random = RandomizedSearchCV(SVC(class_weight="balanced"), random_param, scoring=scoring, refit="f1", cv=5, random_state=state)

    # Model training
    fitted_grid = svc_grid.fit(X_train, Y_train)
    fitted_random = svc_random.fit(X_train, Y_train)

    # Model predictions
    y_random = fitted_random.predict(X_test)
    y_grid = fitted_grid.predict(X_test)

    grid_train_predicted = fitted_grid.predict(X_train)
    random_train_predicted = fitted_random.predict(X_train)

    return fitted_grid, fitted_random, y_grid, y_random, random_train_predicted, grid_train_predicted


def print_score(fitted_grid, fitted_random, Y_test, y_random, y_grid, random_train_predicted, grid_train_predicted,
                Y_train, name=None):
    """ The function prints the scores of the models and the prediction performance """

    # Model comparison
    grid_score = fitted_grid.best_score_
    grid_params = fitted_grid.best_params_
    random_score = fitted_random.best_score_
    random_params = fitted_random.best_params_

    print(f"best grid score {name}:{grid_score}, best grid parameters {name}: {grid_params}")
    print(f"best random scores {name}: {random_score}, best random parameters {name}: {random_params}")

    # Training scores
    random_train_confusion = confusion_matrix(Y_train, random_train_predicted)
    grid_train_confusion = confusion_matrix(Y_train, grid_train_predicted)
    print(f"grid training matrix {name}:{grid_train_confusion}")
    print(f"random training matrix {name}: {random_train_confusion}")

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

    print(f"random confusion matrix {name}: {random_confusion}, grid confusion matrix {name}: {grid_confusion}")
    print(f"random Mathew coefficient {name}: {random_matthews}, grid mathiews coeficient {name}: {grid_matthews}")
    print(f"random accuracy {name}: {random_accuracy}, grid accuracy {name}: {grid_accuracy}")
    print(f"random f1 score {name}: {random_f1}, grid f1 score {name}: {grid_f1}")
    print(f"random precision score {name}: {random_precision}, grid precision score {name}: {grid_precision}")
    print(f"random recall score {name}: {random_recall}, grid recall score {name}: {grid_recall}")

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
            X_train, Y_train, X_test, state=states)

        grid_score, grid_params, grid_confusion, grid_accuracy, grid_f1, grid_precision, grid_recall, grid_matthews, \
        random_score, random_params, random_confusion, random_accuracy, random_matthews, random_f1, random_precision,\
        random_recall, random_train_confusion, grid_train_confusion = print_score(
            fitted_grid, fitted_random, Y_test, y_random, y_grid, random_train_predicted, grid_train_predicted,Y_train, name)

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

    array = numpy.array(metric_list)
    mean = numpy.mean(array, axis=0)

    named_parameters = [parameter_record(*z) for z in parameter_list]
    named_mean = score_record(*mean)
    named_records = [score_record(*y) for y in metric_list]

    return named_mean, model_list, named_parameters, named_records, random_state


def unlisting(named_parameters, named_records):
    """ A function that separates all the scores in independent lists"""
    # Getting all scores random search
    r_mathew = [x.random_matthews for x in named_records]
    r_accuracy = [x.random_accuracy for x in named_records]
    r_f1 = [x.random_f1 for x in named_records]
    r_precision = [x.random_precision for x in named_records]
    r_recall = [x.random_recall for x in named_records]

    # Getting all scores grid search
    g_mathew = [x.grid_matthews for x in named_records]
    g_accuracy = [x.grid_accuracy for x in named_records]
    g_f1 = [x.grid_f1 for x in named_records]
    g_precision = [x.grid_precision for x in named_records]
    g_recall = [x.grid_recall for x in named_records]

    # Hyperparameters grid search
    g_kernel = [y.grid_params["kernel"] for y in named_parameters]
    g_C = [y.grid_params["C"] for y in named_parameters]
    g_gamma = [y.grid_params["gamma"] for y in named_parameters]

    # Hyperparameters random search
    r_kernel = [y.random_params["kernel"] for y in named_parameters]
    r_C = [y.random_params["C"] for y in named_parameters]
    r_gamma = [y.random_params["gamma"] for y in named_parameters]

    return r_mathew, r_accuracy, r_f1, r_precision, r_recall, g_mathew, g_accuracy, g_f1, g_precision, g_recall, \
            g_kernel, g_C, g_gamma, r_kernel, r_C, r_gamma

def plotting(named_parameters, named_records):
    """ Plots everything"""
    sns.set(style='white', context='poster')

    r_mathew, r_accuracy, r_f1, r_precision, r_recall, g_mathew, g_accuracy, g_f1, g_precision, g_recall, \
    g_kernel, g_C, g_gamma, r_kernel, r_C, r_gamma = unlisting(named_parameters, named_records)
    plt.figure()
    plt.subplot(2,2)


# Loading the excel files
global_score = pd.read_excel("sequences.xlsx", index_col=0, sheet_name="global")
local_score = pd.read_excel("sequences.xlsx", index_col=0, sheet_name="local")

# generating X and Y
Y = global_score["label"].copy()
X = global_score.drop(["seq", "label"], axis=1)
robust_X = RobustScaler().fit_transform(X)
standard_X = StandardScaler().fit_transform(X)

# Generate the model and the performance metrics

def scaled_data(X, Y, name):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)
    fitted_grid, fitted_random, y_grid, y_random, random_train_predicted, grid_train_predicted = svc_classification(
        X_train, Y_train, X_test)
    print_score(fitted_grid, fitted_random, Y_test, y_random, y_grid, random_train_predicted, grid_train_predicted,
                Y_train, name)

    return X_train, X_test, Y_train, Y_test
# plotting the data, UMAP transformation
"""
sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
reducer = umap.UMAP(random_state=100, metric="russellrao", n_components=2)
embedding = reducer.fit_transform(X)

plt.scatter(embedding[:,0], embedding[:,1], c=Y, cmap="tab20c") """

# PCA transformation and plotting
"""
pca_r = PCA(n_components=2, random_state=60)
pca_s = PCA(n_components=2, random_state=60)
robust_PCA = pca_r.fit_transform(robust_X)
standard_PCA = pca_s.fit_transform(standard_X)

pd_robust = pd.DataFrame(robust_PCA, columns=["PCA1", "PCA2"])
pd_standard = pd.DataFrame(standard_PCA, columns=["PCA1", "PCA2"])

sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
plt.legend([0,1],prop={'size': 15})
plt.scatter(pd_standard.iloc[:,0], pd_standard.iloc[:, 1], c=Y, cmap="tab20c")
"""
# Different scaling systems
"""
count = 0
L = []
for x in [robust_X, standard_X]:
    count += 1
    X_train, X_test, Y_train, Y_test = scaled_data(x, Y, name=str(count))
    L.append([X_train, X_test, Y_train, Y_test]) """
# Trying the nested CV
s_named_mean, s_model_list, s_named_parameters, s_named_records, s_random_state = mean_nested(standard_X, Y, 0)
# r_named_mean, r_model_list, r_named_parameters = mean_nested(robust_X, Y) --> error