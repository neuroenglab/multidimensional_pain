import pickle
import seaborn as sns
import scipy
from utils_MEM import *
import os, sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import helpers
from statannotations.Annotator import Annotator
from scipy import stats
from paths import DATA_DIR, cohort_results


value=10
plt.rc('font', size=value)          # controls default text sizes
plt.rc('axes', titlesize=value)     # fontsize of the axes title
plt.rc('axes', labelsize=value)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=value)    # fontsize of the tick labels
plt.rc('ytick', labelsize=value)    # fontsize of the tick labels
plt.rc('legend', fontsize=value)


def compute_barplots_CV(performance_path, metric='R2'):
    """
    Compute barplots for CV performance
    :param performance_path: path to the performance files saved as a single file for each CV including all models
    :param metric: metric to be used
    :return: barplots

    """
    files = [ f for f in os.listdir(performance_path) if f.endswith('.pkl') ]
    scores_MEM = {'BayesianRidge' : [ ] , 'GradientSKLEARN' : [ ] , 'HistGradBoost' : [ ] , 'LGBM' : [ ] , 'RF' : [ ] ,
                  'SVR' : [ ] , 'StochsticGradient' : [ ] , 'XGBOOST' : [ ]}
    scores_fixed = {'BayesianRidge' : [ ] , 'GradientSKLEARN' : [ ] , 'HistGradBoost' : [ ] , 'LGBM' : [ ] , 'RF' : [ ] ,
                  'SVR' : [ ] , 'StochsticGradient' : [ ] , 'XGBOOST' : [ ]}
    avperf_MEM = [ ]
    stderrperf_MEM = [ ]

    avperf_fixed = [ ]
    stderrperf_fixed = [ ]


    for idx , file in enumerate(files) :
        with open(performance_path + file , 'rb') as f :
            score = pickle.load(f)
        for model in score.keys() :
            scores_MEM[ model ].append(score[ model ][ "MERF_{}".format(metric) ][ "known" ])
            scores_fixed[ model ].append(score[ model ][ "fixed_{}".format(metric) ][ "known" ])

    for model in scores_MEM.keys() :
        avperf_MEM.append(np.mean(scores_MEM[ model ]))
        stderrperf_MEM.append(stats.sem(scores_MEM[ model ]))
        avperf_fixed.append(np.mean(scores_fixed[ model ]))
        stderrperf_fixed.append(stats.sem(scores_fixed[ model ]))

    models_name = list(scores_MEM.keys())
    x_pos = np.arange(len(models_name))
    ##MEM
    plt.figure()
    fig , ax = plt.subplots()
    ax.bar(x_pos , avperf_MEM , yerr=stderrperf_MEM , align='center' , alpha=0.5 , ecolor='black' , capsize=10)
    ax.set_ylabel(metric)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_name , fontsize=8)
    ax.set_title("MEM")
    ax.yaxis.grid(True)
    plt.savefig(performance_path + "{}_MEM.pdf".format(metric))

    ##FIXED
    plt.figure()
    fig , ax = plt.subplots()
    ax.bar(x_pos , avperf_fixed , yerr=stderrperf_fixed , align='center' , alpha=0.5 , ecolor='black' , capsize=10)
    ax.set_ylabel(metric)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_name , fontsize=8)
    ax.set_title("Fixed")
    ax.yaxis.grid(True)
    plt.savefig(performance_path + "{}_fixed.pdf".format(metric))
    plt.show()

    barplot_all(avperf_MEM , avperf_fixed , metric=metric, results_path=performance_path)
    return avperf_MEM, avperf_fixed


def compute_barplots(results_path, metric='R2'):
    """
    compute barplots for a single model at results path
    """
    models=[]
    R2_score_multi_train=[]
    R2_score_multi=[]
    R2_score_fixed=[]
    R2_score_fixed_train=[]

    with open(results_path + ".pkl", 'rb') as f:
        final_scores = pickle.load(f)

    plt.figure(figsize=(30,20))
    for model in final_scores.keys():
        print(model)
        models.append(model)
        R2_score_multi.append(final_scores[model]["MERF_"+metric]["known"])
        R2_score_fixed.append(final_scores[model]["fixed_"+metric]["known"])

        R2_score_multi_train.append(final_scores[model]["MERF_"+metric]["train"])
        R2_score_fixed_train.append(final_scores[model]["fixed_"+metric]["train"])

    print("R2 multilevel model on valdiation set:")
    print(R2_score_multi)
    print(np.mean(R2_score_multi))

    print("R2 fixed model on valdiation set:")
    print(R2_score_fixed)
    print(np.mean(R2_score_fixed))

    print("wilcoxon Multi vs Fixed: ", scipy.stats.wilcoxon(R2_score_multi, R2_score_fixed))

    X_axis = np.arange(len(models))

    barplot_all(R2_score_multi,R2_score_fixed, metric=metric)
    #
    # plt.bar(X_axis - 0.4, R2_score_multi_train, 0.2, label='Mutlilevel train', color="cornflowerblue")
    # plt.bar(X_axis - 0.2, R2_score_multi, 0.2, label='Mutlilevel val', color="blue")
    # plt.bar(X_axis, R2_score_fixed_train, 0.2, label='Fixed train', color="orange")
    # plt.bar(X_axis + 0.2, R2_score_fixed, 0.2, label='Fixed val', color="red")


    plt.bar(X_axis - 0.3, R2_score_multi, 0.3, label='Mutlilevel val', color="blue")
    plt.bar(X_axis , R2_score_fixed, 0.3, label='Fixed val', color="red")
    plt.xticks(X_axis, models)
    plt.xlabel("models")
    plt.ylabel(metric)
    plt.title(metric+" valdiation set comparison multilvel vs fixed")
    plt.legend()
    plt.savefig(results_path + "_" + metric + ".png")
    plt.savefig(results_path + "_" + metric + ".pdf")
    plt.show()


def barplot_all(R2_score_multi, R2_score_fixed, metric="R2", results_path=None):
    data=pd.DataFrame(index=["multi", "fixed"], data=[R2_score_multi, R2_score_fixed])
    average=np.mean(data, 1)
    std = np.std(data, 1)
    print(average)

    print(scipy.stats.wilcoxon(R2_score_multi, R2_score_fixed))

    data_bar = data.T.melt()
    fig, ax = plt.subplots()
    sns.barplot(x="variable",y="value",data=data_bar)

    pairs = [ ("multi","fixed") ]
    pvals = [ (scipy.stats.wilcoxon(R2_score_multi, R2_score_fixed)).pvalue ]
    # formatted_pvalues = [f"p={p:.2e}" for p in pvals]
    plotting_parameters = {'data': data_bar,'x': 'variable','y': 'value'}
    annotator = Annotator(ax,pairs,**plotting_parameters)

    annotator.set_pvalues(pvals)
    annotator.annotate()
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.yaxis.grid(True)
    plt.savefig(results_path + "_average_" + metric + ".png")
    plt.savefig(results_path + "_average_" + metric + ".pdf")
    plt.show()


def avg_std_bias(performance_path):
    """
    Load subjective bias for all models and CV
    """
    allfiles = []  # Creates an empty list
    subj_bias=pd.DataFrame()
    files = [f for f in os.listdir(performance_path) if f.endswith('.pkl')]
    if len(files) == 0:
        raise Exception("Multilevel results not found. Run main_MEM and try again.")
    for idx, file in enumerate(files):
        with open(performance_path+file, 'rb') as f:
            score = pickle.load(f)
        for model in score.keys():
            subj_bias.loc[:, model+file]=score[model]["subj_bias"]
    return subj_bias

def load_data_psycho(performance_path=RESULTS_DIR_MEM, area="mp", path_data_medication=SUBJECTS_PATH):
    subj_bias=avg_std_bias(performance_path)

    column_patterns_to_remove = ['_AllQuestions', 'rand_', 'PeaksP_', 'PeaksN_', 'height',
                                 '_PreExperiment', '_PostExperiment', '_Before',
                                 'Location_']  # , 'cohort_'
    other_columns_to_remove = ['cohort_HC', 'room_temp_v1', 'room_temp_v2', 'woman_cycle_v1', 'woman_cycle_v2',
                               'time_v1', 'time_v2']


    data_subj = create_subj_dataset(area=area, columns_to_remove=column_patterns_to_remove+other_columns_to_remove, fill_na=False, cohort_path="datasets\\")
    data = data_subj.join(subj_bias.mean(axis=1).rename("average_subj_bias"))
    # data = data[ data["average_subj_bias"].notna() ]
    return data

if __name__=="__main__":
    cohort = "healthy"
    performance_path = cohort_results("Multilevel", cohort)
    subj_bias=avg_std_bias(performance_path)
    avperf_MEM_HC, avperf_fixed_HC=compute_barplots_CV(performance_path , metric='R2')

    cohort = "chronic"
    performance_path = cohort_results("Multilevel", cohort)
    subj_bias=avg_std_bias(performance_path)
    avperf_MEM_CP, avperf_fixed_CP=compute_barplots_CV(performance_path , metric='R2')

    print('Healthy vs chronic')
    print(scipy.stats.wilcoxon(avperf_fixed_CP , avperf_fixed_HC))

    print('Healthy vs chronic MEM')
    print(scipy.stats.wilcoxon(avperf_MEM_CP , avperf_MEM_HC))

