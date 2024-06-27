import matplotlib.pyplot as plt
import pandas as pd
from MultilevelRF import *
from read_multilevel import *
from MEM_regression import *
from utils_MEM import args
from paths import  TRIALS_PATH, RESULTS_DIR_MEM, RESULTS_DIR_CLASS, cohorts
from paths import DATA_DIR, RESULTS_DIR_MEM, FEATURES_DIR, SUBJECTS_PATH, SUBJECTS_PATH, cohort, RESULTS_DIR_BIOMARKERS
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def plot_hist(data, column):
    plt.hist(data[column], alpha=0.5, bins=10, label=column)
    plt.title(column)
    plt.savefig(os.path.join(RESULTS_DIR_MEM, column + ".pdf"))
    plt.show()
    return


def plot_subj_bias_vs_NRS(data, col1="NRS", col2="average_subj_bias"):
    # Create the bar plot for "average_subj_bias" values and overlay on the "NRS" plot
    plt.bar(data.index, data[col2], color='green', alpha=0.4, label=col2)

    # Create the bar plot for "NRS" values and stack it on top of the "average_subj_bias" bars
    plt.bar(data.index, data[col1], bottom=data[col2], color='blue', alpha=0.4, label=col1)

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('{} and {} Stacked Bar Plot'.format(col1, col2))

    # Add a legend
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_MEM, "barplot_overlap_" + col1 + "_" + col2 + ".svg"), format="svg")
    # Show the plot
    plt.show()


def remove_outliers_and_compute_correlation(df , col_target , col_features) :
    df_cleaned = df.copy()

    for col in col_features :
        # Step 1: Identify outliers using the IQR method
        Q1 = df_cleaned[ col ].quantile(0.25)
        Q3 = df_cleaned[ col ].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df_cleaned[ col ] < (Q1 - 1.5 * IQR)) | (df_cleaned[ col ] > (Q3 + 1.5 * IQR))

        # Step 2: Remove outliers
        df_cleaned = df_cleaned[ ~outliers ]

    # Step 3: Compute correlation on the cleaned data and extract p-values
    correlation = df_cleaned[ col_features ].corrwith(df_cleaned[ col_target ])

    p_values = [ pearsonr(df_cleaned[ col_target ] , df_cleaned[ col ])[ 1 ] for col in col_features ]

    return correlation , p_values , df_cleaned


def plot_correlation_features_variables(data , col_target="NRS" , col_features=None) :
    # Call the function to remove outliers and compute correlation with p-values
    num_plots = len(col_features)
    num_rows = 5
    num_cols = 2

    # Split features into groups for multiple subplots (if necessary)
    feature_groups = [ col_features[ i :i + num_rows * num_cols ] for i in range(0 , num_plots , num_rows * num_cols) ]

    for group in feature_groups :
        fig , axs = plt.subplots(min(len(group) , num_rows) , num_cols , figsize=(9 , 15))
        axs = axs.ravel()

        # Call the function to remove outliers and compute correlation with p-values
        correlation , p_values , df_cleaned = remove_outliers_and_compute_correlation(data , col_target , group)

        for i , col in enumerate(group) :
            axs[ i ].scatter(df_cleaned[ col ] , df_cleaned[ col_target ], alpha=0.6, s=20)
            axs[ i ].set_xlabel(col)
            axs[ i ].set_ylabel(col_target)
            # Add correlation value and p-value on top of the plot
            corr = correlation[ i ]
            p_val = p_values[ i ]
            axs[ i ].text(0.5 , 0.9 , f"corr={corr:.2f}, p-value={p_val:.4f}" , transform=axs[ i ].transAxes ,
                          horizontalalignment='center' , verticalalignment='center' , fontsize=10)

            # Try fitting both linear and polynomial regression models and choose the best one
            x = df_cleaned[col].values.reshape(-1, 1)
            y = df_cleaned[col_target].values

            # Fit linear regression
            lin_reg = LinearRegression()
            lin_reg.fit(df_cleaned[col].values.reshape(-1, 1), df_cleaned[col_target].values)
            lin_pred = lin_reg.predict(df_cleaned[col].values.reshape(-1, 1))
            lin_r2 = r2_score(df_cleaned[col_target].values, lin_pred)
            axs[i].plot(x, lin_pred, color='red', linewidth=2, label='Linear Regression')
            # Add R-squared value as text to the plot
            axs[i].text(0.5, 0.8, f"R2={lin_r2:.2f}", transform=axs[i].transAxes,
                        horizontalalignment='center', verticalalignment='center', fontsize=10)
        # Hide any unused subplots
        for j in range(len(group) , num_rows * num_cols) :
            axs[ j ].axis('off')

        plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR_MEM , "correlation_scatter_regress_{}.pdf".format(col_target)), format="pdf")
    plt.show()

    return
def compute_statistics_between_correlations(data, col_target1="NRS", col_target2="NRS_acute-SB", col_features=None):
    correlation_NRS , p_values , df_cleaned = remove_outliers_and_compute_correlation(data , col_target1 , col_features)
    correlation_PHI , p_values , df_cleaned = remove_outliers_and_compute_correlation(data , col_target2 ,features_select)

    #compute abs of correlations NRS and PHI
    correlation_NRS = np.abs(correlation_NRS)
    correlation_PHI = np.abs(correlation_PHI)

    # Plot the correlations without using a DataFrame
    plt.figure(figsize=(8 , 6))  # Adjust the figure size if needed

    # Bar for 'NRS'
    plt.bar("NRS" , correlation_NRS , yerr=correlation_NRS.std(), color='blue' , label='NRS')
    # Bar for 'PHI'
    plt.bar("PHI" , correlation_PHI , yerr=correlation_PHI.std(), color='green' , label='PHI')
    # Perform Wilcoxon signed-rank test
    wilcoxon_statistic , wilcoxon_p_value = wilcoxon(correlation_NRS , correlation_PHI )
    plt.xlabel('Target')
    plt.ylabel('Correlation')
    plt.title(
        f'Correlation between NRS and PHI\nWilcoxon Test: Statistic = {wilcoxon_statistic}, P-value = {wilcoxon_p_value:.4f}')
    plt.ylim(-1 , 1)  # Set the y-axis limit between -1 and 1
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR_MEM , "statistics_bw_correlation_{}_{}.pdf".format(col_target1, col_target2)) , format="pdf")
    plt.show()
    return

def compute_statistics_between_correlations_3(data, col_targets=("NRS", "NRS_acute-SB", "NRS-chronic-SB"), col_features=None):
    correlation_values = []
    p_values = []
    for col_target in col_targets:
        correlation, p_value, df_cleaned = remove_outliers_and_compute_correlation(data, col_target, col_features)
        correlation = np.abs(correlation)  # Compute absolute correlation values
        correlation_values.append(correlation)
        p_values.append(p_value)

    # Calculate standard errors for the correlation distributions
    correlation_stds = [correlation.std() for correlation in correlation_values]

    # Plot the correlations without using a DataFrame
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

    # Width of each bar
    bar_width = 0.25

    # X positions for each group of bars
    x_positions = np.arange(len(col_targets))

    # Bar for each target column
    for i, col_target in enumerate(col_targets):
        plt.bar(x_positions[i], correlation_values[i], yerr=correlation_stds[i], width=bar_width, label=col_target)

    # Perform Wilcoxon signed-rank tests for each pair of correlations
    wilcoxon_values = []
    for i in range(len(col_targets) - 1):
        for j in range(i + 1, len(col_targets)):

            statistic, p_value = wilcoxon(correlation_values[i], correlation_values[j])
            wilcoxon_values.append((statistic, p_value))
            print(col_targets[ i ] , col_targets[ j ])
            print(statistic, p_value)

    plt.xlabel('Target')
    plt.ylabel('Correlation')
    plt.title('Correlation between Different Targets\nWilcoxon Test Results')
    plt.ylim(0, 1)  # Set the y-axis limit between 0 and 1
    plt.xticks(x_positions, col_targets)
    plt.grid(True)
    plt.legend()

    # Superimpose Wilcoxon results on the plot
    for i, (statistic, p_value) in enumerate(wilcoxon_values):
        plt.text(x_positions[i // (len(col_targets) - 1)] + (bar_width * (i % (len(col_targets) - 1))),
                 0.9 - (0.1 * (i % (len(col_targets) - 1))),
                 f'W = {statistic}\np = {p_value:.4f}',
                 ha='center', va='center', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR_MEM, "statistics_bw_correlations.pdf"), format="pdf")
    plt.show()
    return

if __name__=="__main__":
    print("Psychosocial Features analysis on ", cohort, "cohort")
    data_psy = load_data_psycho(performance_path=RESULTS_DIR_MEM , area="mp" , path_data_medication=SUBJECTS_PATH)
    data_psy[ "NRS_now-SB" ] = data_psy[ "NRS_now" ] - data_psy[ "average_subj_bias" ]
    data_psy[ "NRS_avg4-SB" ] = data_psy[ "NRS_avg4wk" ] - data_psy[ "average_subj_bias" ]

    col_target = "NRS_now-SB"  # NRS_avg4wk, NRS_max4wk, NRS_now, average_subj_bias
    data_psy = data_psy[ data_psy[ col_target ].notna() ]

    data_physio = load_data_y_Z_pain(**cohorts)
    #set the index as index without removing
    data_physio = helpers.select_single_area(data_physio , area=args[ 'area' ])
    data_physio.set_index('id', inplace=True, drop=True)
    data_physio_avg = data_physio.groupby('id').mean()


    #merge data_psy with data_physio based on index
    data_merged = data_physio_avg.merge(data_psy, left_index=True, right_index=True)
    plot_hist(data_physio_avg, "NRS")
    #normalize data_mered averagae-sbu_bias between 0 and 1
    data_merged[ "NRS_acute-SB" ] = data_merged[ "NRS" ] - data_psy[ "average_subj_bias" ]

    # col_norm = "NRS_acute-SB"  # NRS_avg4wk, NRS_max4wk, NRS_now, average_subj_bias
    # data_merged[ col_norm ] = (data_merged[ col_norm ] - data_merged[ col_norm ].min()) / (
    #             data_merged[ col_norm ].max() - data_merged[ col_norm ].min())
    # col_norm = "average_subj_bias"  # NRS_avg4wk, NRS_max4wk, NRS_now, average_subj_bias
    # data_merged[ col_norm ] = (data_merged[ col_norm ] - data_merged[ col_norm ].min()) / (
    #             data_merged[ col_norm ].max() - data_merged[ col_norm ].min())

    plot_subj_bias_vs_NRS(data_merged , col1="NRS_acute-SB" , col2="average_subj_bias")


    col_target="NRS_acute-SB"#    TARGET:     NRS_acute-SB (PHI), average_subj_bias (TIP), NRS (NRS)

    with open(RESULTS_DIR_BIOMARKERS + "./features_selected_MEM.pkl"  , 'rb') as fp :
        features_select = pickle.load(fp)

    plot_correlation_features_variables(data_merged , col_target=col_target , col_features=features_select)

    compute_statistics_between_correlations_3(data_merged , col_targets=("NRS" , "NRS_acute-SB" , "average_subj_bias") ,
                                            col_features=features_select)

