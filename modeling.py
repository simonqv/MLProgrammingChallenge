from os import linesep
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import RANDOM

classifiers = {
    # "K-neighbours": KNeighborsClassifier(),
    # "Decision tree": DecisionTreeClassifier(),
    # "Random forest": RandomForestClassifier(random_state=RANDOM),
    # "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM),
    # "Extremely random forest": ExtraTreesClassifier(random_state=RANDOM),
    # "Adaboost": AdaBoostClassifier(),
    # "Bagging": BaggingClassifier(random_state=RANDOM),
    "MLP": MLPClassifier(max_iter=2000, hidden_layer_sizes=(20, 20)),
    # "SVM (rbf)": SVC(),
    # "SVM (linear)": SVC(kernel="linear"),
    # "SVM (polynomial)": SVC(kernel="poly"),
    "Ridge Classifier": RidgeClassifierCV()}

cross_val = StratifiedKFold(shuffle=True, random_state=RANDOM, n_splits=5)


def try_modeling(df, preprocess, X_train, y_train):
    best_clf = None
    name_best_clf = ""
    best_score = 0

    # print("step 1")
    for name_clf, clf in classifiers.items():
        # print("step 2: before pipeline")
        pipeline = Pipeline(steps=[('preprocessor', preprocess), ('classifier', clf)])
        print("Cross_val_score: ")
        score_cv = np.average(cross_val_score(pipeline, X_train, y_train, cv=cross_val, n_jobs=4))
        # print("step 4: before big print")
        print(f"{name_clf} scored {score_cv} during cross validation")

        if score_cv > best_score:
            best_clf = Pipeline
            name_best_clf = name_clf
            best_score = score_cv
            # print(f"\nin loop Best classifier: {name_best_clf}")

    print(f"\nBest classifier: {name_best_clf}")


def try_params_adaboost(df, preprocess, X, y):
    print("enter")
    ada = AdaBoostClassifier()
    pipeline = Pipeline(steps=[('preprocessor', preprocess), ('ada', ada)])
    param_space = {
        'ada__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30, 50],
        'ada__learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
        'ada__algorithm': ['SAMME', 'SAMME.R']
    }
    print("start the grid search")
    clf = GridSearchCV(pipeline, param_grid=param_space, n_jobs=-1, cv=cross_val)
    clf.fit(X, y)
    print("grid search and fit done\n\n")
    # Best:
    print("Best parameter for MLP: ", clf.best_params_)

    # All:
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def try_params_MLP(df, preprocess, X, y):
    print("enter")
    mlp = MLPClassifier(max_iter=2000)
    pipeline = Pipeline(steps=[('preprocessor', preprocess), ('MLP', mlp)])
    param_space = {
        # 'MLP__max_iter': [100, 200, 2000],
        'MLP__hidden_layer_sizes': [(100,)],
        # 'MLP__activation': ['relu'],
        # 'MLP__solver': ['sgd', 'adam'],
        # 'MLP__alpha': [0.0001, 0.05],
        # 'MLP__learning_rate': ['constant'],
    }
    print("start the grid search")
    clf1 = GridSearchCV(pipeline, param_grid=param_space, n_jobs=-1, cv=cross_val)
    clf1.fit(X, y)
    print("grid search and fit done\n\n")
    # Best:
    print("Best parameter for MLP: ", clf1.best_params_)

    # All:
    means = clf1.cv_results_['mean_test_score']
    stds = clf1.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf1.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    return clf1


def final(df, eval_data, X, y, mlp):

    est = mlp.best_estimator_
    est.fit(X, y)

    predictions = est.predict(eval_data)

    OUT = 'labels.txt'

    with open(OUT,'w') as f:
        for prediction in predictions:
            f.write(prediction + linesep)


