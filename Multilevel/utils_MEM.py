import sys
import shap
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os, sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import helpers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.base import clone
import json
from paths import INIT_DIR, DATA_DIR, RESULTS_DIR_MEM, TRIALS_PATH, SUBJECTS_PATH, SUBJECTAREAS_PATH


filename = os.path.join(INIT_DIR, "Multilevel", "args_MEM.json")
with open(filename) as f:
    args = json.load(f)


def load_data_y_Z_pain(HC=True, LBP=True, CRPS=True, SCI_NP=True):
    data_physio=helpers.read_dataset(TRIALS_PATH)
    data_physio = helpers.select_cohort(data_physio, HC=HC, CRPS=CRPS, LBP=LBP, SCI_NP=SCI_NP, dataset_path=SUBJECTS_PATH)
    data_physio.drop("cohort", axis=1, inplace=True)
    ##select only pain data
    data_physio = data_physio[data_physio.pain == 1]
    data_physio["Z"] = np.ones(shape=(data_physio.shape[0], 1)) # variable for MEM models
    data_physio.fillna(value=0, inplace=True)
    return data_physio


def create_subj_dataset(area="con", columns_to_remove=[], fill_na=True, cohort_path="datasets\\"):
    Subjects = helpers.read_dataset(SUBJECTS_PATH)
    SubjectAreas = helpers.read_dataset(SUBJECTAREAS_PATH)
    Area_mp = SubjectAreas[SubjectAreas.Area == 1]
    Area_mp = Area_mp.drop(columns=['Area'])
    Area_con = SubjectAreas[SubjectAreas.Area == 2]
    Area_con = Area_con.drop(columns=['Area'])

    Area_mp = Area_mp.rename(columns=lambda x: x if x == 'id' else x + '_mp')
    Area_con = Area_con.rename(columns=lambda x: x if x == 'id' else x + '_con')
    # data = helpers.join_non_key(Area_mp, Subjects, 'id')
    # data = helpers.join_non_key(data, Area_con, 'id')
    if area=="con":
        data = helpers.join_non_key(Area_con, Subjects, 'id')  # Control area seems better
    elif area=="mp":
        data = helpers.join_non_key(Area_mp, Subjects, 'id')  # Control area seems better
    else:
        raise NameError('No area was selected. Pass mp or con')

    #data = data[data['id'].isin(id)]
    data.set_index("id", inplace=True)
    data = data.select_dtypes(exclude='O')  # Remove non-numeric fields
    data = helpers.drop_by_pattern(data, columns_to_remove)
    if fill_na:
        data = helpers.standard_scaling(data, fill_na)
    return data


def merge_bias_subj_dataset(data, subj_bias):
    id_ = list(subj_bias.index)
    data = data.loc[id_, :]
    data = data.join(subj_bias)
    data.dropna(inplace=True)
    y_train = data.subj_bias
    X_train = data.drop("subj_bias", axis=1)
    return X_train, y_train


def VIF_analysis(data):
    #### TODO how to deal with NaNs

    data.dropna(inplace=True)
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(data.values, i)
                       for i in range(len(data.columns))]

    print(vif_data)


def compute_shap(model, X):
    """
    compute shap value on the multilevel model (fixed components)
    """
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    explainer = shap.TreeExplainer(model.trained_fe_model)
    print("Computing shap values...")
    shap_values = explainer.shap_values(X)
    print("Done.")
    # summarize the effects of all the features
    shap.summary_plot(shap_values, X)
    print("Plotting.")
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="violin", show=False, plot_size=(20,20))
    plt.savefig(RESULTS_DIR_MEM + 'shap_violin_multi_b_id.pdf')
    plt.savefig(RESULTS_DIR_MEM + 'shap_violin_multi_b_id.png')
    plt.show()
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, plot_size=(20,30), max_display=X.shape[1])
    plt.savefig(RESULTS_DIR_MEM + 'shap_bar_multi_b_id.pdf')
    plt.savefig(RESULTS_DIR_MEM + 'shap_bar_multi_b_id.png')


def shap_CVKFold(clf, data,  n_splits=10, random_state=0):
    # from https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a\n"
    # TODO check if reasonable and improve because it is ugly code
    shap.initjs()
    rkf = KFold(n_splits=n_splits , shuffle=True, random_state=random_state)
    list_shap_values = list()
    list_test_sets = list()
    for train_index, test_index in rkf.split(data):
        # X_train , X_test = data.iloc[ train_index ] , data.iloc[ test_index ]
        # y_train , y_test = labels.iloc[ train_index ] , labels.iloc[ test_index ]

        X_train=data.loc[train_index,:]
        X_train.set_index('id', inplace=True, drop=False)
        X_test = data.loc[ test_index , : ]
        X_test.set_index('id' , inplace=True, drop=False)
        y_train=X_train.NRS
        y_test = X_test.NRS
        Z_train = np.ones(shape=(X_train.shape[0], 1))
        Z_test = np.ones(shape=(X_test.shape[0], 1))
        clusters_train = X_train.id
        clusters_test = X_test.id

        X_train.drop(["NRS", "id", "Z"], axis=1, inplace=True)
        X_test.drop(["NRS", "id", "Z"], axis=1, inplace=True)

        # training model
        merfNew = clf
        merfNew.fit(X_train , Z_train , clusters_train , y_train)
        explainer = shap.TreeExplainer(merfNew.trained_fe_model)
        shap_values = explainer.shap_values(X_test)
        # summarize the effects of all the features
        # for each iteration we save the test_set index and the shap_values
        list_shap_values.append(shap_values)
        list_test_sets.append(test_index)

    # combining results from all iterations
    test_set = list_test_sets[ 0 ]
    shap_values = np.array(list_shap_values[ 0 ])
    for i in range(1 , len(list_test_sets)) :
        test_set = np.concatenate((test_set , list_test_sets[ i ]) , axis=0)
        shap_values = np.concatenate((shap_values , list_shap_values[ i ]) , axis=0)
    X_test = data.iloc[ test_set ]
    return X_test , shap_values
