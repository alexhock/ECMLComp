import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from eval import eval_func

def vt(X, Y, valid_X):

    # Loading some example data
    iris = datasets.load_iris()
    X = iris.data[:, [0,2]]
    y = iris.target

    # Training classifiers
    clf1 = ExtraTreesClassifier(n_estimators = 100, criterion = 'gini'),
    clf2 = KNeighborsClassifier(n_neighbors=13)
    clf3 = RandomForestClassifier(n_jobs=5, n_estimators=200, criterion='entropy')
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2, 1, 2])

    clf1 = clf1.fit(X,y)
    clf2 = clf2.fit(X,y)
    clf3 = clf3.fit(X,y)
    eclf = eclf.fit(X,y)

    ##http://scikit-learn.org/stable/modules/ensemble.html#votingclassifier

    #with gridsearch

    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)


    #In order to predict the class labels based on the predicted class-probabilities (scikit-learn estimators in the VotingClassifier must support predict_proba method):
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    #Optionally, weights can be provided for the individual classifiers:
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2,5,1])



def run_vt_final(train_X, train_Y):

    clf1 = ExtraTreesClassifier(n_jobs=3, n_estimators = 100, criterion = 'gini')
    clf2 = KNeighborsClassifier(n_neighbors=13)
    clf3 = RandomForestClassifier(n_jobs=3, n_estimators=200, criterion='entropy')
    wgts = [1, 3]
    print ("voting weights: {0}".format(wgts))
    eclf = VotingClassifier(estimators=[('et', clf1), ('rf', clf3)], voting='hard')
    #eclf = VotingClassifier(estimators=[('et', clf1), ('knn', clf2), ('rf', clf3)], voting='soft', weights=wgts)
    #eclf = VotingClassifier(estimators=[('et', clf1), ('rf', clf3)], voting='soft', weights=wgts)

    eclf = eclf.fit(train_X, train_Y)

    return eclf


if __name__ == '__main__':

    #https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/14335/1st-place-winner-solution-gilberto-titericz-stanislav-semenov/79599

    #C:\Users\AlexH\projects\xgb\NetCla\NetCla\data|baseline
    #E:\xgb\NetCla\NetCla\data
    root_folder = "E:/xgb/NetCla/NetCla/data/"
    #C:\Users\ah14aeb\Projects\ECMLComp\data\mini_data
    #root_folder = "C:/Users/ah14aeb/Projects/ECMLComp/data/mini_data/"

    train_file = root_folder + '/train.csv'
    train_targets_file = root_folder + '/train_target.csv'
    validation_file = root_folder + '/valid.csv'
    validation_target_file = root_folder + '/valid_target.csv'

    train_X = np.loadtxt(train_file, skiprows=1, delimiter='\t')
    train_Y = np.loadtxt(train_targets_file, dtype=np.int)
    print("loaded train data")
    valid_X = np.loadtxt(validation_file, skiprows=1, delimiter='\t')
    valid_Y = np.loadtxt(validation_target_file, dtype=np.int)

    print("loaded valid data")

    #valid_predictions = run_knearest_gridsearch(np.array(train_X, copy=True), np.array(train_Y, copy=True),
    #                                            np.array(valid_X, copy=True), np.array(valid_Y, copy=True))

    #nnk = run_knearest(train_X, train_Y)
    #valid_predictions = nnk.predict(valid_X)

    #http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#example-feature-selection-plot-rfe-with-cross-validation-py
    #http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination

    eclf = run_vt_final(train_X, train_Y)
    valid_predictions = eclf.predict(valid_X)

    eval_func.eval(valid_Y, valid_predictions)
    print "f1: {0}".format(eval_func.get_f1_macro(valid_Y, valid_predictions))
