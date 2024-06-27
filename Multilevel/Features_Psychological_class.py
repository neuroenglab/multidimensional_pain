import matplotlib.pyplot as plt
import pandas as pd
from MultilevelRF import *
from read_multilevel import *
from MEM_regression import *
from scipy.stats import shapiro, mannwhitneyu, ttest_ind, kruskal, wilcoxon
from paths import DATA_DIR, RESULTS_DIR_MEM, FEATURES_DIR, SUBJECTS_PATH, SUBJECTS_PATH, cohort, cohorts
from utils_psychological import create_classes, mapping_col_to_fun
from utils_MEM import args
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

print("Psychosocial Features analysis on ", cohort, "cohort")
# Function to calculate R-squared for each target-predictor pair using training data
def calculate_r2(data, col_target, col_predictor):
    model = LinearRegression()
    model.fit(data[col_target].values.reshape(-1, 1), data[col_predictor])
    y_pred_train = model.predict(data[col_target].values.reshape(-1, 1))

    r2_train = r2_score(data[col_predictor], y_pred_train)

    return r2_train


# Function to train and evaluate a model for each target-predictor pair using training data
def train_classification_model(data, col_target, col_predictor):
    print(col_predictor)
    model = LogisticRegression(random_state=42, class_weight='balanced')
    # model=SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
    X_train = data[col_target].values.reshape(-1, 1)
    label_encoder = LabelEncoder()
    data[ col_predictor ] = data.apply(mapping_col_to_fun(col_predictor),axis=1, col=col_predictor)
    y_train = label_encoder.fit_transform(data[col_predictor])
    # # Cross-validation using StratifiedKFold to compute average accuracy and F1 score
    # cv = StratifiedKFold(n_splits=5 , shuffle=True , random_state=42)
    # # Cross-validation to compute average accuracy and F1 score
    scores_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    scores_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    scores_roc_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    # Compute average accuracy and F1 score
    accuracy = np.mean(scores_accuracy)
    f1 = np.mean(scores_f1)
    roc_auc = np.mean(scores_roc_auc)

    return accuracy, f1, roc_auc



if __name__=="__main__":
    data = load_data_psycho(performance_path=RESULTS_DIR_MEM , area="mp" , path_data_medication=SUBJECTS_PATH)
    data_physio = load_data_y_Z_pain(**cohorts)
    # set the index as index without removing
    data_physio = helpers.select_single_area(data_physio , area=args[ 'area' ])
    data_physio.set_index('id' , inplace=True , drop=True)
    data_physio_avg = data_physio.groupby('id').mean()
    # merge data_psy with data_physio based on index
    data = data.merge(data_physio_avg[ "NRS" ] , left_index=True , right_index=True)
    data[ "NRS_now-SB" ] = data[ "NRS_now" ] - data[ "average_subj_bias" ]
    data[ "NRS_avg4wk-SB" ] = data[ "NRS_avg4wk" ] - data[ "average_subj_bias" ]
    data[ "NRS_acute-SB" ] = data[ "NRS" ] - data[ "average_subj_bias" ]
    cols = [ "HADS_D" , "PCS" , "QST_HPT_mp" , "sick_leave" , "health", "NRS_avg4wk" , 'NRS_now']
    cols_target = [ "NRS" , "average_subj_bias" , "NRS_acute-SB" ] #    TARGET:     NRS_acute-SB (PHI), average_subj_bias (TIP), NRS (NRS)
    # List to store the results
    results_list = [ ]
    # Loop over each target column and each predictor column to calculate R-squared values
    for col_target in cols_target :
        data_tmp = data[pd.notna(data[col_target])]
        for col_predictor in cols :
            data_tmp2=data_tmp[pd.notna(data_tmp[col_predictor])]
            accuracy , f1, roc_auc_score = train_classification_model(data_tmp2 , col_target , col_predictor)
            results_list.append({"col_target" : col_target , "col_predictor" : col_predictor , "accuracy" : accuracy, "f1" : f1, "roc_auc_score" : roc_auc_score})
    # Create a DataFrame from the results_list
    metric="accuracy"
    results_df = pd.DataFrame(results_list)
    plt.figure(figsize=(10 , 6))
    sns.boxplot(data=results_df , x="col_target" , y=metric )
    plt.xlabel("Target Column")
    plt.ylabel(metric)
    plt.title("Models by col_target and col_predictor")
    plt.legend(title="col_predictor" , bbox_to_anchor=(1.05 , 1) , loc='upper left')
    plt.savefig(os.path.join(RESULTS_DIR_MEM, "{}_psychosocial_boxplot.pdf".format(metric)))
    plt.show()
    print(results_df.groupby("col_target").mean())
    pval=wilcoxon(results_df[results_df["col_target"]=="NRS_acute-SB"][metric],results_df[results_df["col_target"]=="average_subj_bias"][metric])
    print("PHI vs TIP", "pval",pval)
    pval=wilcoxon(results_df[results_df["col_target"]=="NRS_acute-SB"][metric],results_df[results_df["col_target"]=="NRS"][metric])
    print("PHI vs NRS", "pval",pval)
    pval=wilcoxon(results_df[results_df["col_target"]=="average_subj_bias"][metric],results_df[results_df["col_target"]=="NRS"][metric])
    print("TIP vs NRS", "pval",pval)

