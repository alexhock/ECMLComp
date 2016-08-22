__author__ = 'alexh'
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from eval import eval_func


def run_train_valid(train_X, train_Y, valid_X, n_trees=10):

      # Our level 0 classifiers
    clfs = [
        RandomForestClassifier(n_estimators = n_trees, criterion = 'gini', n_jobs=3),
        ExtraTreesClassifier(n_estimators = n_trees * 2, criterion = 'gini', n_jobs=3)#,
        #GradientBoostingClassifier(n_estimators = n_trees),
    ]

    # Pre-allocate the data
    layer_0_predictions = np.zeros((train_Y.shape[0], len(clfs)))

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (j)

        clf.fit(train_X, train_Y)

        valid_predict_Y = clf.predict(valid_X)
        layer_0_predictions[:, j] = valid_predict_Y


    # layer 1
    bclf = LogisticRegression()
    bclf.fit(layer_0_predictions, train_Y)

    return bclf.predict(valid_X)



def run(train_X, train_Y, valid_X, valid_Y):
    #X = np.array([ i[:-1] for i in data ], dtype=float)
    #Y = np.array([ i[-1] for i in data ])

    X = train_X
    Y = train_Y

    # The DEV SET will be used for all training and validation purposes
    # The TEST SET will never be used for training, it is the unseen set.
    dev_cutoff = len(Y) * 4/5
    X_dev = X[:dev_cutoff]
    Y_dev = Y[:dev_cutoff]
    X_test = X[dev_cutoff:]
    Y_test = Y[dev_cutoff:]

    n_trees = 10
    n_folds = 5

    # Our level 0 classifiers
    clfs = [
        RandomForestClassifier(n_estimators = n_trees, criterion = 'gini', n_jobs=3)#,
        #ExtraTreesClassifier(n_estimators = n_trees * 2, criterion = 'gini', n_jobs=3)#,
        #GradientBoostingClassifier(n_estimators = n_trees),
    ]

    # Ready for cross validation
    skf = list(StratifiedKFold(Y_dev, n_folds))

    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers

    print 'X_test.shape = %s' % (str(X_test.shape))
    print 'blend_train.shape = %s' % (str(blend_train.shape))
    print 'blend_test.shape = %s' % (str(blend_test.shape))

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)

            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]

            clf.fit(X_train, Y_train)

            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_cv)
            blend_test_j[:, i] = clf.predict(X_test)
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)

    print 'Y_dev.shape = %s' % (Y_dev.shape)

    # Start blending!
    bclf = LogisticRegression()
    bclf.fit(blend_train, Y_dev)

    return bclf, blend_test, Y_test



def do_score(bclf, blend_test, Y_test):
        # Predict now
    Y_test_predict = bclf.predict(blend_test)
    f1_macro_score = metrics.f1_score(Y_test, Y_test_predict, average='macro')
    score = metrics.accuracy_score(Y_test, Y_test_predict)
    print 'Accuracy = %s  F1_macro = %s' % (score, f1_macro_score)
    return score

if __name__ == '__main__':

    #https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14335/1st-place-winner-solution-gilberto-titericz-stanislav-semenov/79599

    #C:\Users\AlexH\projects\xgb\NetCla\NetCla\data|baseline
    root_folder = "C:/Users/alexh/Projects/xgb/NetCla/NetCla/mini_data/"

    train_file = root_folder + '/train.csv'
    train_targets_file = root_folder + '/train_target.csv'
    validation_file = root_folder + '/valid.csv'
    validation_target_file = root_folder + '/valid_target.csv'

    train_X = np.loadtxt(train_file, skiprows=1, delimiter='\t')
    train_Y = np.loadtxt(train_targets_file, dtype=np.int)

    valid_X = np.loadtxt(validation_file, skiprows=1, delimiter='\t')
    valid_Y = np.loadtxt(validation_target_file, dtype=np.int)


    #bclf, blend_test, Y_test = run(train_X, train_Y, valid_X, valid_Y)
    #Y_test_predict = bclf.predict(blend_test)

    valid_predictions = run_train_valid(train_X, train_Y, valid_X)

    eval_func.eval(valid_X, valid_predictions)
    f1_macro = eval_func.get_f1_macro(valid_Y, valid_predictions)
    print 'Best score = %s' % f1_macro