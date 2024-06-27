import pandas as pd
from MultilevelRF import *
from read_multilevel import *
import os, sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
import helpers
import json
from paths import DATA_DIR, RESULTS_DIR_MEM

def mapping_col_to_fun(col):
    #create a dictionary with the mapping between the column name and the function to apply
    mapping = {"HADS_D": discretize_HADS_2,
               "HADS_A": discretize_HADS_2,
                "PCS": discretize_PCS,
                "QST_HPT_mp": discretize_QST,
               "NRS_avg4wk": discretize_pain,
               "NRS_max4wk": discretize_pain,
               "NRS_now": discretize_pain,
               "life_qual" : discretize_life_qual,
               "health":    discretize_health,
               "sick_leave": discretize_sick_leave,
               "MAIA_Noticing": discretize_MAIA,
              "MAIA_NotDistracting" : discretize_MAIA ,
               "MAIA_NotWorrying": discretize_MAIA ,
               "MAIA_AttentionRegulation": discretize_MAIA ,
               "MAIA_EmotionalAwareness": discretize_MAIA ,
               "MAIA_SelfRegulation": discretize_MAIA,
               "MAIA_BodyListening": discretize_MAIA,
               "MAIA_Trusting": discretize_MAIA,
               'sleep': discretize_sleep,
               'PSEQ': discretize_PSEQ,
               'BMI' : discretize_BMI,
               }
    return mapping[col]

def create_classes(row , neg_thresh=-0.5 , pos_thresh=0.5) :
    if row.average_subj_bias <= neg_thresh :
        return -1
    elif row.average_subj_bias > pos_thresh :
        return 1
    else :
        return 0


def discretize_HADS(row , col=[ "HADS_D" ]) :
    if row[ col ] < 8 :
        return "low"
    elif (row[ col ] >= 8) & (row[ col ] < 11) :
        return "medium"
    elif row[ col ] >= 11 :
        return "high"
    else :
        return row[ col ]


def discretize_HADS_2(row , col=[ "HADS_D" ]) :
    if row[ col ] < 8 :
        return "low"
    elif (row[ col ] >= 8) :
        return "high"
    else :
        return row[ col ]


def discretize_PCS(row, col=[ "PCS" ]) :
    if row[ "PCS" ] < 30 :
        return "low"
    elif row[ "PCS" ] >= 30 :
        return "high"
    else :
        return row[ "PCS" ]


def discretize_QST(row,     col=[ "QST_HPT_mp" ]) :
    if row[ "QST_HPT_mp" ] < 0 :
        return "hyper"
    elif row[ "QST_HPT_mp" ] >= 0 :
        return "hypo"
    else :
        return row[ "QST_HPT_mp" ]

def discretize_PSEQ(row , col=[ "pseq" ]) :
    if row[ col] <=12 :
        return "bad"
    elif row[ col] >12 :
        return "good"
    else :
        return row[ col]

def discretize_IPAQ(row , col=[ "ipaq" ]) :
    #TODO
    if row[ col] <=12 :
        return "bad"
    elif row[ col] >12 :
        return "good"
    else :
        return row[ col]


def discretize_sleep(row,   col=[ "sleep" ]) :
    if row[ "sleep" ] < 50 :
        return "good"
    elif row[ "sleep" ] >= 50 :
        return "bad"
    else :
        return row[ "sleep" ]


def discretize_pain(row , col=[ "pain_avg4wk" ]) :
    if row[ col ] < 5 :
        return "low"
    elif row[ col ] >= 5 :
        return "high"
    else :
        return row[ col ]


def discretize_life_qual(row,   col=[ "life_qual" ]) :
    if row[ "life_qual" ] <= 3 :
        return "good"
    elif row[ "life_qual" ] > 3 :
        return "bad"
    else :
        return row[ "life_qual" ]


def discretize_health(row,  col=[ "health" ]) :
    if row[ "health" ] <= 3 :
        return "good"
    elif row[ "health" ] > 3 :
        return "bad"
    else :
        return row[ "health" ]


def discretize_sick_leave(row,  col=[ "sick_leave" ]) :
    if row[ "sick_leave" ] == 0 :
        return "no sick"
    elif row[ "sick_leave" ] == 1 :
        return "high sick"
    else :
        return row[ "health" ]

def discretize_MAIA(row , col=[ "MAIA_Noticing" ]) :

    thresholds={ "MAIA_Noticing" : 3.34,
                 "MAIA_NotDistracting" : 2.06,
                 "MAIA_NotWorrying" : 2.52,
                 "MAIA_AttentionRegulation" : 2.84,
                 "MAIA_EmotionalAwareness" : 3.44,
                 "MAIA_SelfRegulation" : 2.78,
                 "MAIA_BodyListening" : 2.20,
                 "MAIA_Trusting" : 3.37,}
    if row[ col ] < thresholds[col] :
        return "bad"
    elif row[ col ] >= thresholds[col] :
        return "good"
    else :
        return row[ col ]

def discretize_BMI(row , col=[ "BMI" ]) :
    if row[ col ] <= 25 :
        return "good"
    elif row[ col ] > 25 :
        return "bad"
    else :
        return row[ col ]

def load_data_psychological_TIP(performance_path=None, area="mp", path_data_psycho_2=None, HC=True, LBP=True, CRPS=True, SCI_NP=True) :
        subj_bias = avg_std_bias(performance_path)

        column_patterns_to_remove = [ '_AllQuestions' , 'rand_' , 'PeaksP_' , 'PeaksN_' , 'height' , '_PreExperiment' ,
                                      '_PostExperiment' , '_Before' , 'Location_' ]  # , 'cohort_'
        other_columns_to_remove = [ 'room_temp_v1' , 'room_temp_v2' , 'woman_cycle_v1' ,
                                    'woman_cycle_v2' , 'time_v1' , 'time_v2' ]

        data_subj = create_subj_dataset(area=area ,
                                        columns_to_remove=column_patterns_to_remove + other_columns_to_remove ,
                                         fill_na=False , cohort_path="datasets\\")
        # data_psycho_2 = pd.read_csv(path_data_psycho_2)
        # data_psycho_2.set_index("id" , drop=True , inplace=True)
        # data_subj = data_subj.join(data_psycho_2)

        #add TIP
        data = data_subj.join(subj_bias.mean(axis=1).rename("average_subj_bias"))
        data[ "id" ] = data.index
        data = helpers.undummy(data , "cohort")
        data.dropna(subset=[ 'cohort' ] , inplace=True)
        if HC == False :
            data = data[ data.cohort_HC != 1 ]
        if LBP == False :
            data = data[ data.cohort_LBP != 1 ]
        if CRPS == False :
            data = data[ data.cohort_CRPS != 1 ]
        if SCI_NP == False :
            data = data[ data.cohort_SCI_NP != 1 ]
        data = data[ data[ "average_subj_bias" ].notna() ]
        return data

if __name__=="__main__":
    filename = "args_MEM.json"
    with open(filename) as f:
        args = json.load(f)
    cohort="chronic"
    data=load_data_psychological_TIP(performance_path=RESULTS_DIR_MEM , area="mp" ,
                                  path_data_psycho_2=SUBJECTS_PATH, HC=False, LBP=True, CRPS=True, SCI_NP=True)

