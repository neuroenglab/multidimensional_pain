from utils_MEM import *
from MEM_regression import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
from merf.merf import MERF
from merf.viz import plot_merf_training_stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
import os, sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, TRIALS_PATH, SUBJECTS_PATH
import utils_MEM
import pandas as pd
import helpers


scoring_multilabel = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, zero_division=0, average='macro'),  # zero_division to avoid warnings in notebook
           'recall': make_scorer(recall_score, zero_division=0, average='macro'),
           'f1_score': make_scorer(f1_score, average='macro'),
           }
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, zero_division=0),  # zero_division to avoid warnings in notebook
           'recall': make_scorer(recall_score),
           'f1_score': make_scorer(f1_score),
           'roc_auc_score': make_scorer(roc_auc_score)
           }

n_estimators = 60
random_seed=1234
# Models to train
models = {'Decision Tree': DecisionTreeClassifier(max_depth=None,
                                                  random_state=random_seed),
          'Random Forest': RandomForestClassifier(n_estimators=n_estimators,
                                                  random_state=random_seed),
          'Random Forest 2': RandomForestClassifier(n_estimators=100,random_state=random_seed),
          'Extremely Randomized Trees': ExtraTreesClassifier(n_estimators=n_estimators,
                                                             random_state=random_seed),
        'XGBOOST' : xgb.XGBClassifier()
          }


def linear_regression_psycho_bias(X_train, y_train):
    """
    Linear regression to fit and predict the subjective bias
    """
    X = sm.add_constant(X_train)  # adding a constant

    model = sm.OLS(y_train, X).fit()
    predictions = model.predict(X)
    print(mean_squared_error(y_train,predictions,squared=False))
    print_model = model.summary()
    print(print_model)

    helpers.plot_regress_coeff_and_p(model)


def round_subj_bias(row, neg_thresh=-0.5, pos_thresh=0.5):
    if row.subj_bias<=neg_thresh:
        return -1
    elif row.subj_bias>pos_thresh:
        return 1
    else :
        return 0


def classification_cv_psycho_bias(data, labels, models=models, scorers=None, multilabel=True):
    """
    cross validation of models with Kfold. return scores
    """
    scores = {}
    if multilabel:
        cv=KFold(n_splits=5)
        if scorers == None:
            scorers = scoring_multilabel
    else:
        cv=StratifiedKFold(n_splits=3)
        if scorers == None:
            scorers = scoring
    for model_name in sorted(models):
        #clf = OneVsRestClassifier(models[model_name]) needed if classifier does not support directly multi class
        clf = models[model_name]
        score = cross_validate(clf, data, labels, cv=cv.split(data, labels), scoring=scorers)
        scores[model_name] = score
    helpers.print_scores_cv(scores)
    return scores

