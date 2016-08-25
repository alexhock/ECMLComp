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
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from eval import eval_func
from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import PredefinedSplit
from sklearn import svm
from sklearn import preprocessing

def run_svm(train_X, train_Y):
    clf=svm.SVC()
    print clf.get_params().keys()
    clf.fit(train_X, train_Y)
    return clf



def run_rf(train_X, train_Y):
    rf = RandomForestClassifier(n_jobs=5, n_estimators=200, criterion='entropy')
    rf.fit(train_X, train_Y)

    return rf

def run_extratrees(train_X, train_Y):
    # was 10 max features
    clf1 = ExtraTreesClassifier(n_jobs=8, n_estimators = 100, criterion = 'gini', max_features=15)
    clf1.fit(train_X, train_Y)

    return clf1

def run_gridsearch_validation_gen(train_X, train_Y, valid_X, valid_Y, classy, classy_params):
    f1_macro_scorer = metrics.make_scorer(eval_func.get_f1_macro)
    # create test validation split
    train_mask = np.ones(train_Y.shape[0])*-1 # all training samples -1
    valid_mask = np.zeros(valid_X.shape[0]) # all validation samples 0
    train_validation_mask = np.concatenate([train_mask, valid_mask])

    comp_X = np.concatenate([train_X, valid_X])
    comp_Y = np.concatenate([train_Y, valid_Y])

    ps = PredefinedSplit(test_fold=train_validation_mask)

    grid = GridSearchCV(classy, classy_params, scoring=f1_macro_scorer, cv=ps, verbose=10, n_jobs=5)
    grid.fit(comp_X, comp_Y)

    print grid.best_score_
    print grid.best_params_
    best_clf = grid.best_estimator_

    return best_clf, best_clf.predict(valid_X)

def run_lg(train_X, train_Y):
    lg = LogisticRegression()
    lg.fit(train_X, train_Y)
    return lg

def run_naivebayes(train_X, train_Y):
    nb = GaussianNB()
    nb.fit(train_X, train_Y)
    return nb

def run_multinomial_naivebayes(train_X, train_Y):
    nb = MultinomialNB()
    nb.fit(train_X, train_Y)
    return nb


def run_naivebayes_gridsearch(train_X, train_Y, valid_X, valid_Y):
    nb = GaussianNB()
    print nb.get_params().keys()
    nb_params = {
        '': []
    }
    return run_gridsearch_validation_gen(train_X, train_Y, valid_X, valid_Y, nb, nb_params)



def run_knearest_gridsearch(train_X, train_Y, valid_X, valid_Y):

    neigh = KNeighborsClassifier(n_jobs=2)
    print neigh.get_params().keys()
    neigh_params = {
        #'n_neighbors': [10,15,20,25],
        #'weights': ['uniform', 'distance']
        'n_neighbors': [11, 12, 13, 14, 15, 16, 17, 18],
        'weights':['distance']
    }

    return run_gridsearch_validation_gen(train_X, train_Y, valid_X, valid_Y, neigh, neigh_params)

def run_knearest(train_X, train_Y):
    neigh = KNeighborsClassifier(n_jobs=5, weights='distance', n_neighbors=13)
    neigh.fit(train_X, train_Y)
    return neigh

def run_gridsearch_validation(train_X, train_Y, valid_X, valid_Y):

    f1_macro_scorer = metrics.make_scorer(eval_func.get_f1_macro)

    # create test validation split
    train_mask = np.ones(train_Y.shape[0])*-1 # all training samples -1
    valid_mask = np.zeros(valid_X.shape[0]) # all validation samples 0
    train_validation_mask = np.concatenate([train_mask, valid_mask])

    comp_X = np.concatenate([train_X, valid_X])
    comp_Y = np.concatenate([train_Y, valid_Y])

    ps = PredefinedSplit(test_fold=train_validation_mask)

    rf = RandomForestClassifier(n_jobs=2)
    print rf.get_params().keys()
    rf_params = {
        'criterion': ['entropy', 'gini'],
        'n_estimators': [10, 50, 100, 200, 300]
    }

    grid = GridSearchCV(rf, rf_params, scoring=f1_macro_scorer, cv=ps, verbose=10, n_jobs=5)
    #grid.fit(train_X, train_Y)
    grid.fit(comp_X, comp_Y)

    print grid.best_score_
    print grid.best_params_
    best_clf = grid.best_estimator_

    return best_clf.predict(valid_X)

def run_gridsearch_xgb(train_X, train_Y, valid_X):

    f1_macro_scorer = metrics.make_scorer(eval_func.get_f1_macro)

    # setup parameters for xgboost
    #param = {}
    # use softmax multi-class classification
    #param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    #param['eta'] = 0.2
    #param['max_depth'] = 10
    #param['silent'] = 1
    #param['nthread'] = 20
    #param['num_class'] = 43

    gbm = xgb.XGBClassifier(nthread=4)
    print gbm.get_params().keys()
    gbm_params = {
        'learning_rate': [0.05, 0.1],
        'n_estimators': [300, 600],
        'max_depth': [2, 3, 10, 15],
    }

    cv = StratifiedKFold(train_Y)
    grid = GridSearchCV(gbm, gbm_params, scoring=f1_macro_scorer, cv=cv, verbose=10, n_jobs=4)

    grid.fit(train_X, train_Y)

    print grid.best_score_
    print grid.best_params_
    best_clf = grid.best_estimator_

    return best_clf.predict(valid_X)

def run_gridsearch(train_X, train_Y, valid_X):


    f1_macro_scorer = metrics.make_scorer(eval_func.get_f1_macro)

    rf = RandomForestClassifier(n_jobs=2)
    print rf.get_params().keys()
    rf_params = {
        'criterion': ['entropy', 'gini'],
        'n_estimators': [10, 50, 100, 200, 400]
    }

    cv = StratifiedKFold(train_Y)
    grid = GridSearchCV(rf, rf_params, scoring=f1_macro_scorer, verbose=10, n_jobs=5) #cv=cv,

    grid.fit(train_X, train_Y)

    print grid.best_score_
    print grid.best_params_
    best_clf = grid.best_estimator_

    return best_clf.predict(valid_X)

def run_basic_ensemble_avg(train_X, train_Y, valid_X, n_trees=10):

    # Our level 0 classifiers
    clfs = [
        RandomForestClassifier(n_estimators = n_trees, criterion = 'entropy', n_jobs=5),
        ExtraTreesClassifier(n_estimators = n_trees * 2, criterion = 'gini', n_jobs=3),
        GradientBoostingClassifier(n_estimators = n_trees)#,
    ]

    # Pre-allocate the data
    layer_0_predictions = np.zeros((train_Y.shape[0], len(clfs)), dtype=np.int)

    # For each classifier, we train
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % j
        clf.fit(train_X, train_Y)

    # for each classifier predict validation set
    for j, clf in enumerate(clfs):
        print 'Prediction [%s]' % j
        layer_0_predictions[:, j] = clf.predict(valid_X)

    # majority vote
    print("Calculating majority vote")
    results = []
    for i in range(len(layer_0_predictions)):
        sample_predictions = layer_0_predictions[i]
        maj = np.bincount(sample_predictions).argmax()
        results.append(maj)

    return results

def run_train_valid(train_X, train_Y, valid_X, n_trees=10):

      # Our level 0 classifiers
    clfs = [
        RandomForestClassifier(n_estimators = n_trees, criterion = 'entropy', n_jobs=3),
        ExtraTreesClassifier(n_estimators = n_trees * 2, criterion = 'entropy', n_jobs=3)#,
        #GradientBoostingClassifier(n_estimators = n_trees),
    ]

    # Pre-allocate the data
    layer_0_predictions_train = np.zeros((train_Y.shape[0], len(clfs)))
    layer_0_predictions = np.zeros((train_Y.shape[0], len(clfs)))

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (j)

        clf.fit(train_X, train_Y)
        layer_0_predictions_train[:, j] = clf.predict(train_X)

        # store for later
        valid_predict_Y = clf.predict(valid_X)
        layer_0_predictions[:, j] = valid_predict_Y


    # layer 1
    bclf = LogisticRegression()
    bclf.fit(layer_0_predictions_train, train_Y)

    valid_predictions = bclf.predict(layer_0_predictions)

    return valid_predictions



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

    # Predict now
    #Y_test_predict = bclf.predict(blend_test)
    #score = metrics.accuracy_score(Y_test, Y_test_predict)
    #print 'Accuracy = %s' % (score)

    return bclf, blend_test, Y_test

def feature_selection_rfe(train_X, train_Y):

    f1_macro_scorer = metrics.make_scorer(eval_func.get_f1_macro)

    from sklearn.feature_selection import RFECV

    estimator = RandomForestClassifier(n_jobs=6, n_estimators=200, criterion='entropy')
    selector = RFECV(estimator, step=1, cv=5, scoring=f1_macro_scorer)
    selector = selector.fit(train_X, train_Y)
    #selector.support_
    #array([ True,  True,  True,  True,  True,
    #        False, False, False, False, False], dtype=bool)
    print("selector mask: {0}".format(selector.support_))
    print("ranking: {0}".format(selector.ranking_))
    #array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    return selector.support_, selector.ranking_

def feature_selection_ext_trees(train_X, train_Y, valid_X):

    from sklearn.feature_selection import SelectFromModel

    clf = ExtraTreesClassifier()
    clf = clf.fit(train_X, train_Y)
    print("Feature importances: {0}".format(clf.feature_importances_))

    model = SelectFromModel(clf, prefit=True)
    train_X_new = model.transform(train_X)
    valid_X_new = model.transform(valid_X)

    print ("new train shape: {0}  new valid shape: {1}".format(train_X_new.shape, valid_X_new.shape))
    return train_X_new, valid_X_new

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
    #E:\xgb\NetCla\NetCla\data
    root_folder = "E:/xgb/NetCla/NetCla/data/"
    #root_folder = "C:/Users/ah14aeb/Projects/ECMLComp/data/data/"

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

    #valid_predictions = run_train_valid(train_X, train_Y, valid_X)

    #valid_predictions = run_gridsearch_validation(train_X, train_Y, valid_X, valid_Y)

    #nbe, valid_predictions = run_naivebayes_gridsearch(train_X, train_Y, valid_X, valid_Y)
    #nbe = run_naivebayes(train_X, train_Y)
    #valid_predictions = nbe.predict(valid_X)

    #esvm = run_svm(train_X, train_Y)
    #valid_predictions = esvm.predict(valid_X)

    #mbe = run_multinomial_naivebayes(train_X, train_Y)
    #valid_predictions = mbe.predict(valid_X)

    #lg = run_lg(train_X, train_Y)
    #valid_predictions = lg.predict(valid_X)

    ss = preprocessing.StandardScaler(with_mean=True, with_std=True, copy=True)
    ss.fit(train_X)
    #wihtout 0.8543
    #0.8564 0.8558
    s_train_X = ss.transform(train_X)
    s_valid_X = ss.transform(valid_X)
    # ext trees 0.8518 maxfeat10     0.8553 max feat 15 and standarisation

    #sm_train_X, sm_valid_X = feature_selection_ext_trees(s_train_X, train_Y, valid_X)
    feature_mask, feature_ranking= feature_selection_rfe(train_X, train_Y)
    sm_train_X = train_X[:, feature_mask]
    sm_valid_X = valid_X[:, feature_mask]

    trf = run_rf(sm_train_X, train_Y)
    valid_predictions = trf.predict(sm_valid_X)

    #valid_predictions = run_knearest_gridsearch(np.array(train_X, copy=True), np.array(train_Y, copy=True),
    #                                            np.array(valid_X, copy=True), np.array(valid_Y, copy=True))

    #nnk = run_knearest(train_X, train_Y)
    #valid_predictions = nnk.predict(valid_X)

    #http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#example-feature-selection-plot-rfe-with-cross-validation-py
    #http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination

    eval_func.eval(valid_Y, valid_predictions)
    print "f1: {0}".format(eval_func.get_f1_macro(valid_Y, valid_predictions))

    #valid_predictions = run_basic_ensemble_avg(train_X, train_Y, valid_X, n_trees=20)
    #eval_func.eval(valid_Y, valid_predictions)
    #f1_macro = eval_func.get_f1_macro(valid_Y, valid_predictions)
    #print 'Best score = %s' % f1_macro