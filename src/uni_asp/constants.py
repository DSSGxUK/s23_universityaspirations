import os
from enum import Enum


class HyperParameterChoice(Enum):
    RECOMMENDED = "recommended"
    RETUNE = "retune"
    PREV_TUNED = "prev_tuned"


HYPERPARAMETER_CHOICE = HyperParameterChoice.RECOMMENDED
COHORTS_TO_TRAIN = [4, 5]
COHORT_TO_PREDICT_ON = [8]

ROOT_DATA_DIR = "/files/DSSG/ul_data"

RAW_DIR = os.path.join(ROOT_DATA_DIR, "01_raw")
PROCESSED_DIR = os.path.join(ROOT_DATA_DIR, "02_processed")
MODELLING_DIR = os.path.join(ROOT_DATA_DIR, "03_modelling")
# SANITY_CHECKS_PATH = os.path.join(ROOT_DATA_DIR, "04_sanity_plots")
ANALYSIS_DIR = os.path.join(ROOT_DATA_DIR, "04_analysis")

DATAFRAME_PATH = os.path.join(PROCESSED_DIR, "full_df.parquet")
SAVED_MODEL_PATH_DEST = os.path.join(MODELLING_DIR, "saved_models", "lgb_dest_ks4.txt")
SAVED_MODEL_PATH_MAT = os.path.join(MODELLING_DIR, "saved_models", "lgb_gcse_mat.txt")
SAVED_MODEL_PATH_ENG = os.path.join(MODELLING_DIR, "saved_models", "lgb_gcse_eng.txt")
FINAL_CSV_PATH = os.path.join(
    MODELLING_DIR,
    "final_csvs",
    f"coh_{'_'.join([str(x) for x in COHORT_TO_PREDICT_ON])}_results",
)
FINAL_PARAMS_PATH_DEST = os.path.join(MODELLING_DIR, "best_params", "dest_ks4.json")
FINAL_PARAMS_PATH_MAT = os.path.join(MODELLING_DIR, "best_params", "mat_y11.json")
FINAL_PARAMS_PATH_ENG = os.path.join(MODELLING_DIR, "best_params", "eng_y11.json")
FINAL_ENCODERS_PATH_DEST = os.path.join(MODELLING_DIR, "encoders", "dest_ks4.pkl")
FINAL_ENCODERS_PATH_MAT = os.path.join(MODELLING_DIR, "encoders", "mat_y11.pkl")
FINAL_ENCODERS_PATH_ENG = os.path.join(MODELLING_DIR, "encoders", "eng_y11.pkl")


PARAMS_BINARY = {
    "max_depth": 9,
    "learning_rate": 0.017,
    "n_estimators": 340,
    "bagging_fraction": 0.769,
    "bagging_freq": 2,
    "feature_fraction": 0.854,
    "metric": ["auc", "cross_entropy"],
    "objective": "binary",
    "boosting": "gbdt",
    "verbose": -1,
    "threshold": 0.694,
}
PARAMS_MULTI_MAT = {
    "max_depth": 17,
    "learning_rate": 0.033,
    "n_estimators": 139,
    "bagging_fraction": 0.579,
    "bagging_freq": 8,
    "feature_fraction": 0.462,
    "metric": "multi_logloss",
    "objective": "multiclass",
    "boosting": "gbdt",
    "verbose": -1,
    "num_class": 3,
}
PARAMS_MULTI_ENG = {
    "max_depth": 15,
    "learning_rate": 0.019,
    "n_estimators": 228,
    "bagging_fraction": 0.462,
    "bagging_freq": 4,
    "feature_fraction": 0.626,
    "metric": "multi_logloss",
    "objective": "multiclass",
    "boosting": "gbdt",
    "verbose": -1,
    "num_class": 3,
}

COLS_TO_TRAIN_ON = [
    "upn",
    "repeat",
    "transfer",
    "gender",
    "is_white",
    "eal",
    "sen",
    "in_care",
    "premium",
    "suspensions",
    "expulsions",
    "local_auth",
    "north_south",
    "admissions_pol",
    "type",
    "phase",
    "ofstedrating",
    "cluster",
    "mat_y7_pct",
    "eng_y7_pct",
    "mat_y8_pct",
    "eng_y8_pct",
]

CATEGORICAL_FEATURES = [
    "gender",
    "is_white",
    "premium",
    "eal",
    "sen",
    "in_care",
    "transfer",
    "repeat",
    "phase",
    "type",
    "local_auth",
    "north_south",
    "cluster",
    "admissions_pol",
    "ofstedrating",
]
LABELS_SIXTH_FORM = {
    "Sixth form - other - local",
    "Sixth form - current",
    "6th Form Current",
    "6th Form College",
    "A Levels / BTEC / Other equivalent academic route",
    "6th Form Other",
    "Sixth form - other - out of area",
}
HYPERCUBE_POINTS = 1000
BETA = 0.5
HYPERPARAMETERS_LIST = {
    "max_depth": [5, 20],
    "learning_rate": [0.01, 0.2],
    "n_estimators": [50, 1000],
    "bagging_fraction": [0.4, 1],
    "bagging_freq": [1, 10],
    "feature_fraction": [0.25, 1],
    "threshold": [0, 1],
}
FIXED_HYPERPARMETERS_BINARY = {
    "metric": ["auc", "cross_entropy"],
    "objective": "binary",
    "boosting": "gbdt",
    "verbose": -1,
}
FIXED_HYPERPARMETERS_MULTI = {
    "metric": "multi_logloss",
    "objective": "multiclass",
    "boosting": "gbdt",
    "verbose": -1,
    "num_class": 3,
}
INT_HYPERPARAMS = {"max_depth", "n_estimators", "bagging_freq"}
