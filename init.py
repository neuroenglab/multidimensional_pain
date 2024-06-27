import os
from pathlib import Path
from os.path import abspath
import sys
import tkinter as tk
from tkinter import filedialog
import warnings


def check_folder(path, warn=False):
    if not os.path.exists(path):
        if warn:
            warnings.warn("The folder datasets is not in the pain_data folder")
        os.makedirs(path)
        print(f"Created folder {path}")


def check_file(path):
    if not os.path.exists(os.path.join(path)):
        raise NameError(f"The file {path} was not found. Please extract the required data in pain_data folder.")


def check_all(FEATURES_DIR, TRIALS_PATH, SUBJECTS_PATH, RESULTS_DIR, RESULTS_DIR_EDA, RESULTS_DIR_MEM, RESULTS_DIR_CLASS, RESULTS_DIR_BIOMARKERS):
    check_folder(FEATURES_DIR, True)  #check if in the pain_data folder there is a folder called datasets, otherwise raise a warning and create the folder
    check_file(TRIALS_PATH)
    check_file(SUBJECTS_PATH)
    check_folder(RESULTS_DIR)
    check_folder(RESULTS_DIR_EDA)
    check_folder(RESULTS_DIR_MEM)
    check_folder(RESULTS_DIR_CLASS)
    check_folder(RESULTS_DIR_BIOMARKERS)


if __name__ == "__main__":
    # ask the user to select the data folder
    if not os.path.exists("pain_data"):
        print("Please select the path to the pain_data folder")
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory()
        os.chdir(path)
        print("The path is now: ", os.getcwd())
        root.destroy()
        path_data=os.getcwd()
        path_data=path_data+os.sep


    # savepath data ina dataDir.txt file
    INIT_DIR = Path(abspath(__file__)).parent
    with open(os.path.join(INIT_DIR, 'dataDir.txt'), "w") as f:
        f.write(path_data)


    from paths import FEATURES_DIR, TRIALS_PATH, SUBJECTS_PATH, RESULTS_DIR, RESULTS_DIR_EDA, RESULTS_DIR_MEM, RESULTS_DIR_CLASS
    check_all(FEATURES_DIR, TRIALS_PATH, SUBJECTS_PATH, RESULTS_DIR, RESULTS_DIR_EDA, RESULTS_DIR_MEM, RESULTS_DIR_CLASS)

