Repository to replicate the results presented in **Unravelling the physiological and psychosocial signatures of pain by machine learning**

Please refer to:

[![DOI](https://zenodo.org/badge/820861502.svg)](https://zenodo.org/doi/10.5281/zenodo.12568973)

## Requirements
Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation). If using Miniconda, check the "Add Miniconda3 to my PATH envorinment variable" during the installation.

Create a new environment with the requried packages. To do so open a terminal in "path/to/folder/pain" and run: `conda env create -f environment.yml python=3.9.16`

Before running any Python code, activate the virtual environment with `conda activate venvpain` or set it as Python environment if using an IDE.

We recommend [VS Code](https://code.visualstudio.com/) with the default [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-pythonthon) to view and execute the source code.
After creating the `conda` environment, it can be set as current Python environment by pressing `Ctrl+Shift+P` and running `Python: Select interpreter`.

## Data
Run: `init` and select the parent directory where you have extracted all the data. The data must be extracted in a **subfolder called datasets**

Alternately, set it manually by writing the path in a dataDir.txt file placed in the root code folder.

### Data Structure

*Trials.pkl* --> physiological features (SSRH, SSRF, EEG)

*Subjects.csv* --> subjects demogaphic, psychosocial and clinical information

*SubjectAreas.csv* --> information relative to the painful area

## Code

### Global parameters

#### `args_global.json`
- cohort: which cohorts to be selected, either "healthy" (HC: Healthy cohort), or "chronic" (LBP: low back pain, CRPS: complex regional apin syndrome, SCI_NP: spinal cord injury with neuropathic pain).
- Other parameters specify the relative path of input and output files and folder. Their value should not be changed.

### EDA

Exploratory data analysis with jupyter notebook.

1. `Correlation_analysis` : correlation heatmap for feature reduction considering subject, area and trial specific features independetnly. VIF analysis to remove remaining collinear features

2. `EDA`: explore features

3. `demographic` and `info demo paper`: to extract demographic and other information

### Classification 

Classification scripts. Starting from the csv with extracted features, classification of pain vs baseline

Set parameters in `args_class.json` and run `main_classification`.

#### `args_class.json`
- flags: to set true and false depending on which framework step are to be performed. feat_select (true: perform correlation step and save features in a pkl file), grid_search (grid_search over hyperparameter grid and save best model in a pkl file), LOSO (perform LOSO classification and save results in a pkl file), ROCAUC (compute and plot roc auc). The first time main_classification is run, it is necessary to set them to true. Afterwards, they can be set to false to only visualize the results previously trained. 

- signal: which signal to use in the classification (EEG, SCH,SCF, true/false). If multisignal, set all to true.

- feat_selection: parameters for the feature selection step. corr_threshold to be used, method (as in pandas corr (pearson, spearman, kendall)), drop_feat_path (file name where to save features to drop).

- gridsearch: gridsearch_path (file name where to save features to drop).

- LOSO: loso_path (file name where to save classification results), metric (sklearn metric selected to plot barplots of performance).

- XAI: shap_flag_XGB/feature_importance_flag_XGB/shap_RF (true/false, if and which model to use to compute shap values and if compute intrinsic XGB feature importance), selectfeat_path (where to save the most important shap features), selectfeat_num (how many most important shap feature to select).

#### Other files
- `loso_classify`: Leave one subject one classification using multiprocessing for boosting computational time. different models and different metrics are considered. Scoring for binary and scoring_multilabel for mutlilabel classification. 

- `helpers`: functions to read data, join different datasets, compute shap, print scores.

- `feature_selection`: utils to compute correlation and save features correlated to be dropped.

- `shap_helpers`: helpers to apply shap and return shap values.

- `grid_search_models`: utils to perform hyperparameter tuning.

- `read_results`: read pickle file from binary classification and plot results.

### Multilevel
Multilevel vs Fixed regression of subjective reported pain level (NRS). Comparison of the two approaches and psychological assessments.

Set parameters in `args_MEM.json` and run `main_MEM` (`main_classification` must have already been run).

#### `args_MEM.json`
- area: area (mp).

- scale: (true/false, it true apply standard scaling).

- signal: which signal to use in the classification (EEG, SCH,SCF, true/false). If multisignal, set all to true.

- regression: parameters for the regression. max_iterations(max iteations for the iterative MEM approaches), metric (R2/RMSE, metric to plot).

- XAI: shap (true/false, whether to compute shap values on the fixed part of MEM models), shap_model (which base regressor to use, e.g., RF for random forest regressor), selectfeat_path (where to load the most important shap features as selected during the classification step).

- save: save_results_path (where to save subjective bias and metrics from regression, both fixed and MEM).

#### Other files

- `MEM_regression`: functions to compute MEM and fixed regression, returing data and metrics.

- `read_multilevel`: utils to read and plot performance obtained during training and testing of fixed and MEM regressor.

- `utils_MEM`: utils to LOAD and process data.

- `Features_Psychological`: analysis of psychosocial and clinical traits with respect to subjetive bias (i.e., mismatch between subjective reported pain level and physiological activation).

- `Features_Psychological_class`: classification of each psychosocial and clinical traits from TIP, NRS, and PHI. results are plotted and compared.

- `Psycho_vs_Physio`: Computing pearson correlation between NRS, TIP and PHI and physiological biomarkers. Results are plotted and compared.

### Preprocessing

Preprocessing scripts. Starting from the original raw data these scripts preprocess the signals and extract the relevant physiological and psychosocial features.
