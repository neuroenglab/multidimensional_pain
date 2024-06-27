import os
import json
from pathlib import Path
from init import check_all


def join_path(relPath, dir=True):
    global DATA_DIR
    if dir:
        return os.path.join(DATA_DIR, relPath) + os.path.sep
    else:
        return os.path.join(DATA_DIR, relPath)


def cohort_results(folder, cohort):
    global RESULTS_DIR
    return os.path.join(RESULTS_DIR, folder, cohort) + os.path.sep


INIT_DIR = Path(os.path.abspath(__file__)).parent
dataDirPath = os.path.join(INIT_DIR, 'dataDir.txt')
if not os.path.isfile(dataDirPath):
    raise Exception('Data directory not set. Run init.py to set it.')
with open(dataDirPath) as f:
    DATA_DIR = f.readline().rstrip('\n')
filename = os.path.join(INIT_DIR, 'args_global.json')
with open(filename) as f:
    dataPaths = json.load(f)
cohort = dataPaths["cohort"]

if cohort == "healthy":
    cohorts = {"HC":True,"LBP":False, "CRPS":False, "SCI_NP":False}
elif cohort == "chronic":
    cohorts = {"HC":False,"LBP":True, "CRPS":True, "SCI_NP":True}
else:
    raise Exception("Cohort must be either healthy or chronic")

RESULTS_DIR = join_path(dataPaths["results_dir"])
FEATURES_DIR = join_path(dataPaths["features_dir"])
TRIALS_PATH = os.path.join(FEATURES_DIR, dataPaths["trials_path"])
SUBJECTS_PATH = os.path.join(FEATURES_DIR, dataPaths["subjects_path"])
SUBJECTAREAS_PATH = os.path.join(FEATURES_DIR, dataPaths["subjectareas_path"])

check_all(FEATURES_DIR, TRIALS_PATH, SUBJECTS_PATH, RESULTS_DIR, cohort_results("EDA", "chronic"), cohort_results("Multilevel", "chronic"), cohort_results("Classification", "chronic"), os.path.join(cohort_results("Multilevel", "chronic"), "features_selected"))
check_all(FEATURES_DIR, TRIALS_PATH, SUBJECTS_PATH, RESULTS_DIR, cohort_results("EDA", "healthy"), cohort_results("Multilevel", "healthy"), cohort_results("Classification", "healthy"), os.path.join(cohort_results("Multilevel", "healthy"), "features_selected"))

RESULTS_DIR_EDA = cohort_results("EDA", cohort)
RESULTS_DIR_MEM = cohort_results("Multilevel", cohort)
RESULTS_DIR_CLASS = cohort_results("Classification", cohort)
RESULTS_DIR_BIOMARKERS = os.path.join(RESULTS_DIR_MEM, "features_selected")

check_all(FEATURES_DIR, TRIALS_PATH, SUBJECTS_PATH, RESULTS_DIR, RESULTS_DIR_EDA, RESULTS_DIR_MEM, RESULTS_DIR_CLASS, RESULTS_DIR_BIOMARKERS)