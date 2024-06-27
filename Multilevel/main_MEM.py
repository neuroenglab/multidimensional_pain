from MEM_regression import *
from sklearn.ensemble import RandomForestRegressor
from read_multilevel import *
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, SGDRegressor
from sklearn.svm import SVR
from utils_MEM import args
from paths import  TRIALS_PATH, RESULTS_DIR_MEM, RESULTS_DIR_CLASS, cohorts, RESULTS_DIR_BIOMARKERS
from lightgbm import LGBMRegressor
import os

## LOAD data
data_physio = load_data_y_Z_pain(**cohorts)
data_physio = helpers.select_single_area(data_physio, area=args['area'])
y=data_physio['pain']
groups = data_physio["id"]
random_state=0

selectfeat_path = RESULTS_DIR_CLASS + args['XAI']['selectfeat_path'] + ".pkl"
if not os.path.isfile(selectfeat_path):
    raise Exception(f"{selectfeat_path} not found. Run main_classification first.")

## select the best pain biomarkers as computed with shap
with open(selectfeat_path, 'rb') as fp:
    features_select = pickle.load(fp)
features_select=(features_select.tolist())+['NRS', 'id', 'Z']
print(features_select)
data_physio=data_physio[features_select]

n_splits=10
rkf = KFold(n_splits=n_splits , shuffle=True, random_state=random_state)
save_idx=0
data_physio.reset_index(drop=True , inplace=True)
max_iterations = args['regression']['max_iterations']
models = {"RF" : RandomForestRegressor() ,
            "LGBM" : LGBMRegressor() , "XGBOOST" : XGBRegressor() ,
            "GradientSKLEARN" : GradientBoostingRegressor() , "BayesianRidge" : BayesianRidge() ,
            "StochsticGradient" : SGDRegressor() , "SVR" : SVR() ,
            "HistGradBoost" : HistGradientBoostingRegressor() ,
            }
save_path_multilevel = args['save']['save_results_path']+"_{}".format(save_idx)
if args[ 'regression' ][ 'train_regression' ] :
    for train_index , test_index in rkf.split(data_physio) :
        save_idx=save_idx+1
        ## create data struture for MEM
        X_train=data_physio.loc[train_index,:]
        X_train.set_index('id', inplace=True, drop=False)
        X_known = data_physio.loc[ test_index , : ]
        X_known.set_index('id' , inplace=True, drop=False)
        y_train=X_train.NRS
        y_known = X_known.NRS
        Z_train = np.ones(shape=(X_train.shape[0], 1))
        Z_known = np.ones(shape=(X_known.shape[0], 1))
        clusters_train = X_train.id
        clusters_known = X_known.id
        X_train.drop(["NRS", "id", "Z"], axis=1, inplace=True)
        X_known.drop(["NRS", "id", "Z"], axis=1, inplace=True)

        if args["scale"]:
            scaler = StandardScaler()
            X_train = pd.DataFrame(data=scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_known = pd.DataFrame(data=scaler.transform(X_known), index=X_known.index, columns=X_known.columns)

        try:
            with open(RESULTS_DIR_MEM+save_path_multilevel+".pkl", 'rb') as f:
                final_scores = pickle.load(f)
        except FileNotFoundError:
            final_scores = {}

        for model_name in sorted(models):
            if model_name in final_scores:
                continue
            model = models[model_name]
            ##mixed effect mdoel
            print("********* training multilevel mixed effect model: ***************")
            subj_bias, RMSE, YHAT, R2= model_train_physio(X_train, Z_train, clusters_train, y_train, X_known, Z_known, clusters_known,y_known, merf=MERF(model, max_iterations=max_iterations), plot=False, explain=False, save_path=RESULTS_DIR_MEM+save_path_multilevel+model_name)
            ##FIXED model
            print("********* training fixed effect model: ***************")
            RMSE_fix, YHAT_fix, R2_fixed = train_fixed_model(X_train, X_known, y_train, y_known, model_fix=model)

            scoring_tmp={"subj_bias": subj_bias, "MERF_RMSE": RMSE, "MERF_R2": R2, "fixed_RMSE": RMSE_fix,"fixed_R2": R2_fixed}
            final_scores[model_name]=scoring_tmp
            with open(RESULTS_DIR_MEM+save_path_multilevel+".pkl", 'wb') as f:
                pickle.dump(final_scores, f)

subj_bias=avg_std_bias(RESULTS_DIR_MEM)
avperf_MEM_HC, avperf_fixed_HC=compute_barplots_CV(RESULTS_DIR_MEM , metric=args['regression']['metric'])
avperf_MEM_HC , avperf_fixed_HC = compute_barplots_CV(RESULTS_DIR_MEM , metric="R2")

if args['XAI']['shap']:
    X_test, shap_values=shap_CVKFold(MERF(models[args['XAI']['shap_model']], max_iterations=max_iterations) , data_physio, n_splits=n_splits , random_state=0)
    shap.summary_plot(shap_values , X_test.drop(["NRS", "id", "Z"], axis=1))
    plt.figure()
    shap.summary_plot(shap_values , X_test.drop(["NRS", "id", "Z"], axis=1) , plot_type="violin" , show=False , plot_size=(20 , 20))
    plt.savefig(RESULTS_DIR_MEM+save_path_multilevel+'_shap_violin.pdf')
    plt.show()
    plt.figure()
    shap.summary_plot(shap_values , X_test.drop(["NRS", "id", "Z"], axis=1), plot_type="bar" , show=False , plot_size=(20 , 30) ,
                        max_display=X_test.drop(["NRS", "id", "Z"], axis=1).shape[ 1 ])
    plt.savefig(RESULTS_DIR_MEM+save_path_multilevel+'_shap_bar.pdf')

    col_ascending = X_test.columns[ np.argsort(np.abs(shap_values).mean(0)) ]
    with open(RESULTS_DIR_BIOMARKERS + "./features_selected_MEM.pkl" , 'wb') as fp :
        pickle.dump(col_ascending[ -10 : ] , fp)
