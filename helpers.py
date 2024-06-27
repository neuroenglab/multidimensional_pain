import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from paths import DATA_DIR

mpl.rcParams['pdf.fonttype'] = 42  # Otherwise text in PDF exports is not editable in AI

## HELPERS TO LOAD DATA AND CLEAN
def read_dataset(dataset_path):
    """
    Read datasets. expected format: ".csv" ".pkl"
    dataset_name = relative path in parent DATA_DIR folder of the dataset to be loaded
    """
    if dataset_path is None:
        raise  Exception('No dataset path has been passed')
    
    if not os.path.isfile(dataset_path):
        raise  Exception(dataset_path + ' does not exist')

    # if file format is csv
    if dataset_path.endswith('.csv'):
        data = pd.read_csv(dataset_path)

    # if file format is pickle
    elif dataset_path.endswith('.pkl'):
        data = pd.read_pickle(dataset_path)
    else:
        raise Exception('Dataset format not supported')

    return data


def select_cohort(df, HC, LBP, CRPS, SCI_NP, dataset_path):
    """
    Select in df only the desired cohorts (selected independently).
    df = dataframe to be processed
    dataset_path = file where the correspondence id and specific cohort is saved
    """
    cohorts = read_dataset(dataset_path)
    cohorts=cohorts.loc[:, ["id", "cohort_CRPS", "cohort_HC", "cohort_LBP", "cohort_SCI_NP"]]
    df_final = join_non_key(df, cohorts, "id")
    df_final=undummy(df_final , "cohort")
    df_final.dropna(subset=[ 'cohort' ], inplace=True)
    if HC==False:
        df_final=df_final[df_final.cohort_HC!=1]
    if LBP==False:
        df_final=df_final[df_final.cohort_LBP!=1]
    if CRPS == False:
        df_final = df_final[df_final.cohort_CRPS != 1]
    if SCI_NP == False:
        df_final = df_final[df_final.cohort_SCI_NP != 1]
    return df_final

def select_feature_signals(data, EEG=True, SCH=True, SCF=True):
    """
    Select only the desired signals (selected independently).
    data = dataframe to be processed
    return = dataframe with only the selected physiological signals

    """
    to_drop=[]
    if EEG==False:
        to_drop=to_drop+[col for col in data.columns if "EEG" in col]
    if SCH==False :
        to_drop=to_drop+[col for col in data.columns if "SCH" in col ]
    if SCF==False:
        to_drop=to_drop+[col for col in data.columns if "SCF" in col]
    data.drop(to_drop, axis=1, inplace=True)
    return data

def select_single_area(df, area="mp"):
    """
    Function to select one of the three experimented areas:
    (most painful --> categorical 1, control --> categorical 2, additional --> categorical 3)
    """
    areas_dict = {"mp": 1,
                  "con": 2,
                  "add": 3}
    return df[df["Area"] == areas_dict[area]]

def join_non_key(df1, df2, keyColumns):
    """
    join dataframe from columns names
    """
    return df1.set_index(keyColumns).join(df2.set_index(keyColumns)).reset_index()


def drop_by_pattern(df, patterns):
    """"
    Drop all columns with a specific pattern
    """
    columns_to_drop = [col for col in df.columns if any(feat in col for feat in patterns)]
    return df.drop(columns=columns_to_drop)

def drop_if_exist(df, columns):
    columns = set(columns).intersection(df.columns)
    return df.drop(columns=columns)


### Dummy and undummy costum functions
def dummy_area(data, skipControl=True, skipAd=False):
    data['Area_mp'] = data['Area'] == 1
    if not skipControl:
        data['Area_con'] = data['Area'] == 2  # N.B. collinear to others
    if not skipAd:
        data['Area_ad'] = data['Area'] == 3
    return data.drop(columns='Area')


def undummy(data, feat, drop_flag=False):
    cols = data.columns[data.columns.map(lambda x: x.startswith(feat + '_'))]
    data = data.copy()
    for c in cols:
        value = c[len(feat)+1:]
        data.loc[data[c]>0, feat] = value
    if drop_flag:
        data = data.drop(columns=cols)
    data[feat] = data[feat].astype('category')
    return data


#
def standard_scaling(data, fillna, not_to_scale=[]):
    """
    Standard scaler in selected features. save in a dataframe
    """
    to_scale = ~data.columns.isin(not_to_scale)
    scaledData = StandardScaler().fit_transform(data.loc[:, to_scale])
    #data.loc[:, to_scale] = pd.DataFrame(scaledData, index=data.loc[:, to_scale].index, columns=data.loc[:, to_scale].columns)
    data.loc[:, to_scale] = scaledData
    if fillna:
        data.loc[:, to_scale] = data.loc[:, to_scale].fillna(0)
        return data
    else:
        return data

## helpers to print and visualize classification data

def score(model, X_test, y_test, scorers):
    """
    Compute the scores of the model on the test set

    Parameters
    ----------
    model : sklearn model The model to be evaluated
    X_test : pandas DataFrame The test set
    y_test : pandas Series The test labels
    scorers : list of sklearn scorers The scorers to be used to evaluate the model

    """
    scores = {}
    for scorer in scorers:
        scores[scorer] = scorers[scorer](model, X_test, y_test)
    return scores


def print_scores(scores):
    for key, value in scores.items():
        print('\t\t\t{:^40s}'.format( key))
        print_scores_per_model(value)
        print()

def print_scores_per_model(scores):
    for s in sorted(scores):
        print('\t\t\t\t%-18s%.3f' % (remove_test_(s), scores[s]))

def print_scores_cv(scores):
    for key, value in scores.items():
        print('\t\t\t{:^50s}'.format( key))
        print_scores_cv_per_model(value)
        print()

def print_scores_cv_per_model(scores):
    for s in sorted(scores):
        print('\t\t\t\t%-18s%.3f +/- %.3f' % (remove_test_(s), np.average(scores[s]), np.std(scores[s])))

def remove_test_(text):
    return text[len('test_'):] if text.startswith('test_') else text

def savefig(folderName, plotName):
    plt.tight_layout()  # Fixes subplot size so it fits perfectly in figure
    folderPath = os.path.join(DATA_DIR, 'Plots', folderName)
    os.makedirs(folderPath, exist_ok=True)
    plt.savefig(os.path.join(folderPath, plotName + '.png'))
    plt.savefig(os.path.join(folderPath, plotName + '.pdf'))  # also svg and eps available but pdf seems best


def plot_regress_coeff(coeff, conf_lo, conf_hi,  ax=plt.gca()):
    ax.axvline(linestyle=':', color='k')
    for lower, upper, y in zip(conf_lo, conf_hi, range(len(coeff))):
        ax.plot((lower, upper), (y, y), '-|', color='k')
    ax.scatter(coeff, coeff.index, marker='x', color='r')
    ax.set_xlabel('Coefficient')


def plot_regress_coeff_and_p(model):
    coeff = pd.concat([pd.DataFrame({'coeff': model.params, 'p': model.pvalues}),
                  model.conf_int().rename(columns={0: "conf_lo", 1: "conf_hi"})], axis=1)
    coeff.sort_values('p', inplace=True, ascending=False)

    significant_covariates = coeff.index[coeff['p'] < 0.01].difference(['id Var','Intercept'])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plot_regress_coeff(coeff['coeff'], coeff["conf_lo"], coeff["conf_hi"], ax1)
    ax2.barh(coeff.index, coeff['p'])
    ax2.axvline(0.05, color='k')
    plt.text(0.08, ax2.get_ylim()[1]*0.7,'p = 0.05', rotation=90)
    ax2.set_xlabel('p-value')
    plt.show()
    return significant_covariates

