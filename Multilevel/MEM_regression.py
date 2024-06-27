from MultilevelRF import *
import numpy as np
from merf.merf import MERF
from merf.viz import plot_merf_training_stats
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def train_fixed_model(X_train, X_known, y_train, y_known, X_new=None, y_new=None, model_fix=RandomForestRegressor(n_estimators=300, n_jobs=-1)):
    model_fix.fit(X_train, y_train)

    # predict TRAIN
    y_hat_train_fix = model_fix.predict(X_train)
    rmse_train_fix=mean_squared_error(y_train, y_hat_train_fix, squared=False)
    print("RMSE train sklearn ", rmse_train_fix)
    print("R2 train sklearn ", r2_score(y_train, y_hat_train_fix))

    # KNOWN --> test on data with known patients. we trained their subjective bias
    y_hat_known_fix = model_fix.predict(X_known)
    rmse_known_fix = mean_squared_error(y_known, y_hat_known_fix, squared=False)
    print("RMSE known sklearn ", rmse_known_fix)
    print("R2 known sklearn ", r2_score(y_known, y_hat_known_fix))

    #NEW
    if X_new is not None:
        y_hat_new_fix = model_fix.predict(X_new)
        #pmse_new_fix = np.mean((y_new - y_hat_new_fix) ** 2)
        #rmse_new_fix = np.sqrt(np.sum((y_new - y_hat_new_fix) ** 2)) / len(y_new)
        rmse_new_fix = mean_squared_error(y_new, y_hat_new_fix, squared=False)
        r2_score_new=r2_score(y_new, y_hat_new_fix)
        print("RMSE new sklearn ", rmse_new_fix)
        print("R2 new sklearn ", r2_score(y_new, y_hat_new_fix))
    else:
        #pmse_new_fix=None
        rmse_new_fix = None
        y_hat_new_fix=None
        r2_score_new=None

    RMSE={"train": rmse_train_fix, "known": rmse_known_fix, "new": rmse_new_fix}
    YHAT={"train": y_hat_train_fix, "known": y_hat_known_fix}
    R2={"train": r2_score(y_train, y_hat_train_fix), "known":  r2_score(y_known, y_hat_known_fix), "new": r2_score_new}
    return RMSE, YHAT, R2


def model_train_physio(X_train, Z_train, clusters, y_train, X_known=None, Z_known=None, clusters_known=None, y_known=None,X_new=None, Z_new=None, clusters_new=None, y_new=None, plot=True, merf=MERF(max_iterations=20), explain=True, save_path="fit_merf",**kwargs):
    '''
    model_train_physio train multilevel MERF models and evalaute in known and new data.
    pass X, Z (bias vector) cluter group and y (predictors) for each test to be evaluated.
    If model is not passed, automatically use MERF random forest regressor with 20 iterations. Pass MERF object
    return subj_bias, RMSE, YHAT

    subj_bias: vector with biases for each group specified in clusters
    RMSE dict with RMSE for train, known and new data
    YHAT dict with prediction values
    '''
    merf.fit(X_train, Z_train, clusters, y_train)
    yhat_train = merf.predict(X_train, Z_train, clusters)

    mse_train=mean_squared_error(y_train, yhat_train, squared=False)
    r2_train=r2_score(y_train, yhat_train)
    print("RMSE MERF train sklearn ", mse_train)
    print("R2 MERF train sklearn ", r2_score(y_train, yhat_train))
    # if condition to distinguish when we extract the subjective bias and when we train and test
    if y_known is not None:
        yhat_known = merf.predict(X_known, Z_known, clusters_known)
        mse_known = mean_squared_error(y_known, yhat_known, squared=False)
        r2_known=r2_score(y_known, yhat_known)
        print("RMSE MERF known sklearn ", mse_known)
        print("R2 MERF known sklearn ", r2_score(y_known, yhat_known))
    else:
        mse_known=None
        yhat_known=None
        r2_known=None
    if y_new is not None:
        yhat_new = merf.predict(X_new, Z_new, clusters_new)
        mse_new= mean_squared_error(y_new, yhat_new, squared=False)
        r2_new=r2_score(y_new, yhat_new)
        print("RMSE MERF new sklearn ", mse_new)
        print("R2 MERF new sklearn ", r2_score(y_new, yhat_new))
    else:
        mse_new = None
        yhat_new=None
        r2_new=None

    # subj bias
    subj_bias = merf.trained_b
    subj_bias.rename(columns={0: "subj_bias"}, inplace=True)
    if plot:
        plt.figure()
        plt.plot(y_train, yhat_train, "x")
        plt.xlabel("NRS real")
        plt.ylabel("NRS predicted")
        plt.figure()
        plt.hist(merf.trained_b)
        plt.title("subjective bias distribution")
        plt.show()
        plot_merf_training_stats(merf, num_clusters_to_plot=10)
        plt.savefig(save_path+'MERF_training.pdf')
    if explain:
        ## only with classical methods not shap
        # TODO this only for forest
        importances = (merf.trained_fe_model).feature_importances_
        std = np.std([tree.feature_importances_ for tree in merf.trained_fe_model.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=X_train.columns)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
        # features = [0, 1]
        # plot_partial_dependence(mrf.trained_fe_model, X_known, features)

        compute_shap(merf, X_train)

    RMSE={"train": mse_train, "known": mse_known, "new": mse_new}
    YHAT={"train": yhat_train, "known": yhat_known, "new": yhat_new}
    R2 = {"train": r2_train, "known": r2_known, "new": r2_new}
    return subj_bias, RMSE, YHAT, R2