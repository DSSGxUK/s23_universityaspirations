import json
import logging
import os
import pickle

import lightgbm as lgb
import pandas as pd

from uni_asp.constants import (
    COHORT_TO_PREDICT_ON,
    COLS_TO_TRAIN_ON,
    DATAFRAME_PATH,
    FINAL_CSV_PATH,
    FINAL_ENCODERS_PATH_DEST,
    FINAL_ENCODERS_PATH_ENG,
    FINAL_ENCODERS_PATH_MAT,
    FINAL_PARAMS_PATH_DEST,
    HYPERPARAMETER_CHOICE,
    PARAMS_BINARY,
    SAVED_MODEL_PATH_DEST,
    SAVED_MODEL_PATH_ENG,
    SAVED_MODEL_PATH_MAT,
    HyperParameterChoice,
)
from uni_asp.modelling.training import get_the_Ps, inv_transform

logger = logging.getLogger(__name__)

########################################################################################


def return_dataset(X_input, cohort_to_test, encoder_dict):
    """
    Perform additional LightGBM-specific preprocessing (inference path)

    This differs from the functions used in the training step in two ways:
      1. Only model inputs are returned (the X values)
      2. Pupils with any missing data are dropped

    Args:
        df: Dataframe to preprocess (created by the preprocessing step of the main
            pipeline)
    """

    X_input = X_input[X_input["cohort"].isin(cohort_to_test)]
    X_input = X_input[COLS_TO_TRAIN_ON]

    orig_len = len(X_input)
    X_input = X_input.dropna().reset_index(drop=True)
    num_dropped = orig_len - len(X_input)
    logger.info(
        f"Dropped {num_dropped/orig_len:.0%} of pupils ({num_dropped} / {orig_len}) "
        f"because of missing data"
    )

    X_input_upn = X_input.copy()
    X_input = X_input.drop("upn", axis=1)

    X_input = X_input.copy()
    for col in encoder_dict:
        X_input[col] = encoder_dict[col].transform(X_input[[col]])

    needed_cols = X_input.columns.tolist()
    X_input[needed_cols] = X_input[needed_cols].apply(
        pd.to_numeric, errors="coerce", axis=1
    )

    return pd.concat([X_input, X_input_upn[["upn"]]], axis=1)


def predict(X_input, kind):
    """
    Loads the trained light gbm to make predictions.

    There are three models which can be used to predict three different things. The
    model used is specified by the 'kind' parameter, which can take one of three values:
        - 'dest_ks4' : for ks4 predictions
        - 'mat_gcse' : for GCSE math grades
        - 'eng_gcse' : for English GCSE grades

    Args:
        X_input: The dataframe after model specific preprocessing
        kind: Specifies which model to use and hence what to predict

    Returns:
        A dataframe containing probabilities for the KS4 destination or GCSE grade band
            (low/medium/high) respectively.
    """
    if kind == "dest_ks4":
        multi = False
        model = lgb.Booster(model_file=SAVED_MODEL_PATH_DEST)

        if HYPERPARAMETER_CHOICE == HyperParameterChoice.RECOMMENDED:
            threshold = PARAMS_BINARY["threshold"]
        else:
            with open(FINAL_PARAMS_PATH_DEST) as f:
                hyperparams = json.load(f)
                threshold = hyperparams["threshold"]

    elif kind == "mat_gcse":
        multi = True
        model = lgb.Booster(model_file=SAVED_MODEL_PATH_MAT)
        threshold = None
    elif kind == "eng_gcse":
        multi = True
        model = lgb.Booster(model_file=SAVED_MODEL_PATH_ENG)
        threshold = None
    else:
        raise ValueError(f"Unexpected value for 'kind'. Got {kind!r}.")

    prob_df = get_the_Ps(model, X_input, multi=multi, threshold=threshold)

    return prob_df


########################################################################################


def run_lgbm_inference(kind):
    """
    Run the inference step of the pipeline to generate predictions for the current year
    9 cohort.

    This function will load a trained model and save a csv (& parquet) of probabilities.

    Args:
        kind: Specifies what to predict. Should be one of 'dest_ks4', 'mat_gcse' or
            'eng_gcse'
    """
    df = pd.read_parquet(DATAFRAME_PATH)
    if kind == "dest_ks4":
        with open(FINAL_ENCODERS_PATH_DEST, "rb") as f:
            encoder_dict = pickle.load(f)
    elif kind == "mat_gcse":
        with open(FINAL_ENCODERS_PATH_MAT, "rb") as f:
            encoder_dict = pickle.load(f)
    elif kind == "eng_gcse":
        with open(FINAL_ENCODERS_PATH_ENG, "rb") as f:
            encoder_dict = pickle.load(f)
    else:
        raise ValueError(f"Unexpected value for 'kind'. Got {kind!r}.")

    X_train = return_dataset(
        df, cohort_to_test=COHORT_TO_PREDICT_ON, encoder_dict=encoder_dict
    )

    prob_df = predict(X_train, kind=kind)
    prob_df = inv_transform(prob_df, encoder_dict)

    os.makedirs(FINAL_CSV_PATH, exist_ok=True)
    fpath = os.path.join(FINAL_CSV_PATH, f"model_results_{kind}")
    logger.info(f"Saving model results to {fpath}.parquet")
    prob_df.to_csv(f"{fpath}.csv", index=False)
    prob_df.to_parquet(f"{fpath}.parquet")
