import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import *
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import categorical_encoder as cat_encode

MODELS_TO_RUN = ['RF', 'DT', 'LR', 'SVM', 'KNN']
FEATURES = ['platform', 'visitorType', 'CategoryID']
PATH_RESULTS = "./results/"
PATH_IMAGES = "./images/"


# Simple undersampling of the majority class to ensure balanced data set used for training/validation
def under_sampling(df: pd.DataFrame, response_col: str):
    df_one = df.loc[df[response_col] == 1]
    df_zero = df.loc[df[response_col] == 0]
    if len(df_one) < len(df_zero):
        df_zero = df_zero.sample(n=len(df_one))
    else:
        df_one = df_one.sample(n=len(df_zero))
    return pd.concat([df_zero, df_one])


def prepare_input(df: pd.DataFrame):
    """
    Prepare key inputs for the later steps of the pipeline
    Input:
    - df: pandas dataframe
    Output:
    - train data transformed, test data transformed, original train data, original test data,
    categorical pipeline
    """
    df['NewID'] = df.index
    train_set, test_set = split_train_test_by_id(df, 0.3, 'NewID')
    cat_attribs = [FEATURES[0], FEATURES[1], FEATURES[2]]
    cat_pipeline = Pipeline([
        ('selector', cat_encode.DataFrameSelector(cat_attribs)),
        ('cat_encoder', cat_encode.CategoricalEncoder(encoding="onehot-dense")),
    ])
    train_set_num = train_set[FEATURES]
    train_prepared = cat_pipeline.fit_transform(train_set_num)
    test_set_num = test_set[FEATURES]
    test_prepared = cat_pipeline.transform(test_set_num)
    return train_prepared, test_prepared, train_set, test_set, cat_pipeline


def find_best_model(df: pd.DataFrame, grid_size: str, outcome_var: str, file_name=None):
    """
    Use grid search to find best model
    Input:
    - df: pandas dataframe
    - grid_size: one of 3 possible values: 'test', 'small', 'large'
    - outcome_var: the outcome variable
    - file_name: file name of the csv file containing the results
    Output:
    - either return a dataframe or save results as csv file
    """
    clfs, grid = define_clfs_params(grid_size)

    # define models to run
    models_to_run = MODELS_TO_RUN

    # call clf_loop and store results in results_df
    train_prepared, test_prepared, train_set, test_set, cat_pipeline = prepare_input(df)
    results_df = clf_loop(models_to_run, clfs, grid, train_prepared, test_prepared,
                          train_set[outcome_var], test_set[outcome_var])
    # save to csv
    if file_name:
        file_name = PATH_RESULTS + file_name
        results_df.to_csv(file_name, index=False)
    else:
        return results_df


# Calculate AUC score for 1-feature decision tree as baseline results
def baseline_model(df: pd.DataFrame, outcome_var: str):
    train_prepared, test_prepared, train_set, test_set, cat_pipeline = prepare_input(df)
    dec_tree = DecisionTreeClassifier(max_depth=1, min_samples_split=10)
    y_pred_probs = dec_tree.fit(train_prepared, train_set[outcome_var]).predict_proba(test_prepared)[:, 1]
    print("AUC score of 1-feature decision tree: " + str(roc_auc_score(test_set[outcome_var], y_pred_probs)))


# Fit the best model and plot ROC graph
def fit_random_forest(df: pd.DataFrame, outcome_var: str):
    train_prepared, test_prepared, train_set, test_set, cat_pipeline = prepare_input(df)
    model = RandomForestClassifier(max_depth=5, max_features='sqrt', min_samples_split=2,
                                   n_estimators=10, n_jobs=-1)
    model.fit(train_prepared, train_set[outcome_var])
    model_preds = model.predict_proba(test_prepared)
    prob_true = model_preds[::, 1]
    plot_roc("RandomForest", prob_true, test_set[outcome_var], "save")


# Code from line 109 to line 321 is adapted from Rayid Ghani's github: https://github.com/rayidghani/magicloops
# Plot the ROC curve
def plot_roc(name, probs, true, output_type):
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    if output_type == 'save':
        plt.savefig(PATH_IMAGES + name + '_roc.png')
    else:
        plt.show()


# Generate binary prediction at a specified cutoff point defined as k percent of the sample
# Only apply to y sorted
def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


# Calculate precision at k
def precision_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision


# Calculate recall at k
def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


# Create confusion matrix
def create_confusion_matrix(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    table = confusion_matrix(y_true_sorted, preds_at_k)
    return table


# Plot precision recall curve
def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])

    name = model_name
    plt.title(name)
    if output_type == 'save':
        plt.savefig(name)
    elif output_type == 'show':
        plt.show()
    else:
        plt.show()


def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
            'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
            'LR': LogisticRegression(penalty='l1', C=1e5),
            'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
            'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss="hinge", penalty="l2"),
            'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    large_grid = {
        'RF': {'n_estimators': [1, 10, 100, 1000, 10000], 'max_depth': [1, 5, 10, 20, 50, 100],
               'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10], 'n_jobs': [-1]},
        'LR': {'penalty': ['l1', 'l2'], 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
        'SGD': {'loss': ['hinge', 'log', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet']},
        'ET': {'n_estimators': [1, 10, 100, 1000, 10000], 'criterion': ['gini', 'entropy'],
               'max_depth': [1, 5, 10, 20, 50, 100], 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10],
               'n_jobs': [-1]},
        'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1, 10, 100, 1000, 10000]},
        'GB': {'n_estimators': [1, 10, 100, 1000, 10000], 'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
               'subsample': [0.1, 0.5, 1.0], 'max_depth': [1, 3, 5, 10, 20, 50, 100]},
        'NB': {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10]},
        'SVM': {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
        'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']}
    }

    small_grid = {
        'RF': {'n_estimators': [10, 100], 'max_depth': [5, 50], 'max_features': ['sqrt', 'log2'],
               'min_samples_split': [2, 10], 'n_jobs': [-1]},
        'LR': {'penalty': ['l1', 'l2'], 'C': [0.00001, 0.001, 0.1, 1, 10], 'solver': ['liblinear']},
        'SGD': {'loss': ['log', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet']},
        'ET': {'n_estimators': [10, 100], 'criterion': ['gini', 'entropy'], 'max_depth': [5, 50],
               'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 10], 'n_jobs': [-1]},
        'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1, 10, 100, 1000, 10000]},
        'GB': {'n_estimators': [10, 100], 'learning_rate': [0.001, 0.1, 0.5], 'subsample': [0.1, 0.5, 1.0],
               'max_depth': [5, 50]},
        'NB': {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10]},
        'SVM': {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
        'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']}
    }

    test_grid = {
        'RF': {'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'], 'min_samples_split': [10]},
        'LR': {'penalty': ['l1'], 'C': [0.01], 'solver': ['liblinear']},
        'SGD': {'loss': ['perceptron'], 'penalty': ['l2']},
        'ET': {'n_estimators': [1], 'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],
               'min_samples_split': [10]},
        'AB': {'algorithm': ['SAMME'], 'n_estimators': [1]},
        'GB': {'n_estimators': [1], 'learning_rate': [0.1], 'subsample': [0.5], 'max_depth': [1]},
        'NB': {},
        'DT': {'criterion': ['gini'], 'max_depth': [1], 'min_samples_split': [10]},
        'SVM': {'C': [1], 'kernel': ['linear']},
        'KNN': {'n_neighbors': [5], 'weights': ['uniform'], 'algorithm': ['auto']}
    }

    if grid_size == 'large':
        return clfs, large_grid
    elif grid_size == 'small':
        return clfs, small_grid
    elif grid_size == 'test':
        return clfs, test_grid
    else:
        return 0, 0


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df = pd.DataFrame(columns=('model_type', 'clf', 'parameters', 'auc-roc', 'r_at_5', 'r_at_10', 'r_at_20',
                                       'r_at_30', 'r_at_35', 'r_at_40', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30',
                                       'p_at_35', 'p_at_40'))
    for n in range(1, 2):
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 35.0),
                                                       recall_at_k(y_test_sorted, y_pred_probs_sorted, 40.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 35.0),
                                                       precision_at_k(y_test_sorted, y_pred_probs_sorted, 40.0)]

                except IndexError as e:
                    print('Error:', e)
                    continue
    return results_df


# Create training and test set: code taken from Aurelien Geron's github https://github.com/ageron/handson-ml
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def main():
    df = pd.read_csv('./data/ECommerceDataSet.csv')
    # Remove outliers
    df = df.loc[df['SessionRevenue'] != 500000]
    # Apply under sampling
    df = under_sampling(df, 'CVR')
    # Find the best model
    find_best_model(df, 'small', 'CVR', 'ml_results_after_undersampling.csv')
    # Fit the best model, in this case, random forest model
    fit_random_forest(df, 'CVR')
    # Find AUC for 1-feature decision tree as the baseline results
    baseline_model(df, 'CVR')


if __name__ == "__main__":
    main()
