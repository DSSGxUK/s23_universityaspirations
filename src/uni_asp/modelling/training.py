import json
import logging
import os
import pickle
import warnings
from contextlib import contextmanager
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold

from uni_asp.constants import (
    BETA,
    CATEGORICAL_FEATURES,
    COLS_TO_TRAIN_ON,
    DATAFRAME_PATH,
    FINAL_ENCODERS_PATH_DEST,
    FINAL_ENCODERS_PATH_ENG,
    FINAL_ENCODERS_PATH_MAT,
    FINAL_PARAMS_PATH_DEST,
    FINAL_PARAMS_PATH_ENG,
    FINAL_PARAMS_PATH_MAT,
    FIXED_HYPERPARMETERS_BINARY,
    FIXED_HYPERPARMETERS_MULTI,
    HYPERCUBE_POINTS,
    HYPERPARAMETER_CHOICE,
    PARAMS_BINARY,
    PARAMS_MULTI_ENG,
    PARAMS_MULTI_MAT,
    SAVED_MODEL_PATH_DEST,
    SAVED_MODEL_PATH_ENG,
    SAVED_MODEL_PATH_MAT,
    HyperParameterChoice,
)
from uni_asp.modelling.model_pre import (
    generate_hypercube,
    return_datasets_b,
    return_datasets_m,
)

logger = logging.getLogger(__name__)


@contextmanager
def suppress_lgb_parameter_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        yield


########################################################################################


def get_the_Ps(model, X_input, multi: bool, threshold: Optional[float]):
    """
    Use the model to generate predictions.

    The returned dataframe contains both predicted labels and probability values.
    In the case of multi==True, the predictions are for Low, Medium, High GCSE grades,
    while in the case of multi==False, the predictions are for the KS4 destination.

    Args:
        model: The LGBM model
        X_input: A dataframe containing features of the pupils to predict on
        multi: If True, generate multi-class predictions
        threshold: The threshold to use for classification in the binary case (when
            multi is False)
    """
    if multi == True:
        prob_list = model.predict(X_input.drop("upn", axis=1))
        prob_low = pd.DataFrame(prob_list[:, 0], columns=["prob_low_gcse"])
        prob_med = pd.DataFrame(prob_list[:, 1], columns=["prob_med_gcse"])
        prob_high = pd.DataFrame(prob_list[:, 2], columns=["prob_high_gcse"])
        labels = pd.DataFrame(np.argmax(prob_list, axis=1), columns=["label_gcse"])
        final = pd.concat(
            [X_input.reset_index(drop=True), prob_low, prob_med, prob_high, labels],
            axis=1,
        )

    else:
        probs = model.predict(X_input.drop("upn", axis=1))
        prob_labels = (probs > threshold).astype(float)

        prob_df = pd.concat(
            [
                pd.DataFrame(probs, columns=["prob_dest_ks4"]),
                pd.DataFrame(prob_labels, columns=["label_dest_ks4"]),
            ],
            axis=1,
        )
        prob_df["threshold"] = threshold

        final = pd.concat([prob_df, X_input.reset_index(drop=True)], axis=1)

    return final


def calculate_metrics(
    X_train, X_test, y_train, y_test, params, threshold, multi=False, average="weighted"
):
    """
    Trains a model, calculates metrics on a held-out test set and prints the result.

    The metrics calculated are: F0.5, Accuracy, Precision & Recall

    The 'average' parameter is used in multi class for calculating the F-0.5 metric.

    Args:
        X_train: A dataframe containing the training features and UPN's.
        X_test: A dataframe containing the test features and UPN's.
        y_train: A series containing the outcome variable for the training set.
        y_test: A series containing the outcome variable for the testing_set.
        params: A dictionary of hyper-parameters to use when training the LGBM model.
        threshold: A threshold to use in the case of binary predictions (i.e. KS4
            destinations).
        multi: If True, then calculate metrics for a multi-class model, else calculate
            metrics for a binary model. This should be set to True for GCSE predictions
            and False for KS4 destination predictions.
        average: For metrics on a multi-class prediction problem, this is passed to
            sklearn when calculating the precision, recall and F0.5 score.
    """

    logger.info("Training model (%s) on training set", y_train.name)
    train_data = lgb.Dataset(X_train.drop("upn", axis=1), label=y_train)
    test_data = lgb.Dataset(X_test.drop("upn", axis=1))

    with suppress_lgb_parameter_warning():
        model = lgb.train(train_set=train_data, valid_sets=[test_data], params=params)

    y_probs = model.predict(X_test.drop("upn", axis=1))

    if multi:
        y_pred = np.argmax(y_probs, axis=1)
        fbeta = fbeta_score(y_test, y_pred, average=average, beta=BETA)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
    else:
        y_pred = (y_probs > threshold).astype(int)
        fbeta = fbeta_score(y_test, y_pred, beta=BETA)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

    logger.info(
        f"Model metrics on test set ({y_train.name}): "
        f"F0.5 SCORE = {fbeta:.3f}, "
        f"ACCURACY = {accuracy:.3f}, "
        f"PRECISION = {precision:.3f}, "
        f"RECALL = {recall:.3f}"
    )


def train_gbm(
    X_train,
    y_train,
    X_test,
    y_test,
    outcome_var,
    multi: bool,
    hyperparam_choice: HyperParameterChoice,
):
    """
    Train the LightGBM model with the option of retuning hyper-parameters.

    Args:
        X_train: A dataframe containing the training features and UPN's.
        X_test: A dataframe containing the test features and UPN's.
        y_train: A series containing the outcome variable for the training set.
        y_test: A series containing the outcome variable for the testing_set.
        outcome_var: The target variable (one of 'dest_ks4', 'mat_y11', 'eng_y11')
        multi: If True, then train a multi-class model, else train binary model
        hyperparam_choice: Determines which hyper-parameters to use

    Returns:
        A trained model

    The hyperparam_choice argument has three options:
      - 'recommended': Use the recommended hyperparameters, tuned by the DSSG team
      - 'retune': Re-run the hyper-parameter tuning, and save the result
      - 'prev_tuned': Load hyper-parameters previously generated when running using
            'retune'
    """
    hyperparam_path = {
        "dest_ks4": FINAL_PARAMS_PATH_DEST,
        "mat_y11": FINAL_PARAMS_PATH_MAT,
        "eng_y11": FINAL_PARAMS_PATH_ENG,
    }

    if hyperparam_choice == HyperParameterChoice.RECOMMENDED:
        logger.info("Training with recommended parameters (%s)", outcome_var)
        recommended_params = {
            "dest_ks4": PARAMS_BINARY,
            "mat_y11": PARAMS_MULTI_MAT,
            "eng_y11": PARAMS_MULTI_ENG,
        }
        best_params = recommended_params[outcome_var]
        threshold = best_params.get("threshold")
        best_params = {k: v for k, v in best_params.items() if k != "threshold"}

    elif hyperparam_choice == HyperParameterChoice.RETUNE:
        logger.info("Retuning hyperparameters (%s)", outcome_var)
        best_params, threshold = retune_hyperparams(X_train, y_train, multi)
        if not multi:
            best_params["threshold"] = threshold
        os.makedirs(os.path.dirname(hyperparam_path[outcome_var]), exist_ok=True)
        with open(hyperparam_path[outcome_var], "w") as f:
            json.dump(best_params, f, indent="    ")

    elif hyperparam_choice == HyperParameterChoice.PREV_TUNED:
        logger.info("Loading previously tuned hyperparameters (%s)", outcome_var)
        with open(hyperparam_path[outcome_var]) as f:
            loaded_params = json.load(f)
        threshold = loaded_params.get("threshold")
        best_params = {k: v for k, v in loaded_params.items() if k != "threshold"}

    else:
        raise ValueError(
            f"Unexpected value for hyperparam_choice. Got {hyperparam_choice!r}."
        )

    calculate_metrics(X_train, X_test, y_train, y_test, best_params, threshold, multi)

    logger.info("Retraining model on full dataset")
    final_train_data = lgb.Dataset(
        pd.concat([X_train, X_test]).drop("upn", axis=1).reset_index(drop=True),
        label=pd.concat([y_train, y_test]).reset_index(drop=True),
    )

    with suppress_lgb_parameter_warning():
        model = lgb.train(
            best_params,
            final_train_data,
            categorical_feature=CATEGORICAL_FEATURES,
        )

    return model


def retune_hyperparams(
    X_train, y_train, multi: bool, num_folds=5, n_hypercube_points=HYPERCUBE_POINTS
):
    """
    Retune the model hyper-parameters by searching over a Latin Hyper-cube.

    Each candidate set of hyper-parameters is evaluated using a cross-validation
    estimate of the F0.5 score.

    Args:
        X_train: A dataframe containing training data features
        y_train: A series containing training data targets
        multi: If True, then train a multiclass model, else train a binary model
        num_folds: The number of cross-validation folds
        n_hypercube_points: The number of points in the Latin hypercube

    Returns:
        Tuple: (best_model_params, best_threshold)
    """
    best_params = None
    best_threshold = None
    best_sf = -1

    # Define the k-fold split for the cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    hypercube = generate_hypercube(n_hypercube_points)
    for cnt, params in hypercube.items():
        logger.info(f"Testing hyper-parameters ({cnt}): {params}")
        (
            cv_score,
            full_params,
            threshold,
        ) = evaluate_hyperparameters_with_crossvalidation(
            X_train, y_train, params, kf, multi
        )

        logger.info(f"Average Cross-Validation Fbeta Score: {cv_score:.4f}")

        if cv_score > best_sf:
            best_sf = cv_score
            best_params = full_params
            best_threshold = threshold

    return best_params, best_threshold


def evaluate_hyperparameters_with_crossvalidation(
    X_train, y_train, params: dict, kf: KFold, multi: bool
) -> float:
    """
    Evaluate a set of hyperparameters using K-fold cross validation

    The metric used is the F0.5 score.

    Args:
        X_train: A dataframe containing training data features
        y_train: A series containing training data targets
        params: A dictionary of hyperparameters
        kf: A KFold object defining the K folds in the cross-validation
        multi: If True, then train a multiclass model, else train a binary model

    Returns:
        Tuple: The average F0.5 score over the folds, the parameters used (including any
            additional fixed parameters) and the threshold used
    """

    fixed_params = FIXED_HYPERPARMETERS_MULTI if multi else FIXED_HYPERPARMETERS_BINARY
    threshold = params.get("threshold")
    model_params = {k: v for k, v in params.items() if k != "threshold"}
    model_params.update(fixed_params)

    cv_scores = []
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        logger.debug(f"Training on Fold {fold + 1}...")

        # Split the data into train and validation sets
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Create LightGBM datasets for training and validation
        train_data = lgb.Dataset(X_tr.drop("upn", axis=1), label=y_tr)
        valid_data = lgb.Dataset(
            X_val.drop("upn", axis=1), label=y_val, reference=train_data
        )

        with suppress_lgb_parameter_warning():
            model = lgb.train(
                model_params,
                train_data,
                valid_sets=[valid_data],
                categorical_feature=CATEGORICAL_FEATURES,
                callbacks=[lgb.log_evaluation(200)],
            )

        # Evaluate the model on the validation set
        val_probs = model.predict(X_val.drop("upn", axis=1))
        if multi:
            y_val_labels = np.argmax(val_probs, axis=1)
            score = fbeta_score(y_val, y_val_labels, average="weighted", beta=BETA)
        else:
            y_val_labels = (val_probs > threshold).astype(int)
            score = fbeta_score(y_val, y_val_labels, beta=BETA)
        cv_scores.append(score)

    logger.debug(f"Fbeta scores: {cv_scores}")
    avg_cv_score = sum(cv_scores) / len(cv_scores)
    return avg_cv_score, model_params, threshold


def inv_transform(df, encoder_dict):
    """
    Does inverse transform to convert from ordinal encoding to categories.
    """

    for col in encoder_dict.keys():
        df[col] = encoder_dict[col].inverse_transform(df[col].values.reshape(-1, 1))
    return df


########################################################################################


def run_lgbm_training(outcome_var, save=True):
    """
    Run the full LGBM training step

    This will train one of the three models, depending on the value of outcome_var:
      - 'dest_ks4': KS4 destination (i.e. sixth-form or not)
      - 'mat_y11': GCSE maths (low, medium, high)
      - 'eng_y11': GCSE english (low, medium, high)

    Args:
        outcome_var: The target variable, which determines the model type
        save: If True, then save the probabilities and predicted classes for each pupil
    """
    meta = {
        "dest_ks4": {"save_path": SAVED_MODEL_PATH_DEST, "multi": False},
        "mat_y11": {"save_path": SAVED_MODEL_PATH_MAT, "multi": True},
        "eng_y11": {"save_path": SAVED_MODEL_PATH_ENG, "multi": True},
    }

    if outcome_var not in meta:
        raise ValueError(f"Unexpected value for outcome_var. Got {outcome_var!r}.")

    df = pd.read_parquet(DATAFRAME_PATH)
    if meta[outcome_var]["multi"]:
        X_train, y_train, X_test, y_test, encoder_dict = return_datasets_m(
            df,
            outcome_var,
            categorical_features=CATEGORICAL_FEATURES,
            cols_to_train=COLS_TO_TRAIN_ON,
        )
    else:
        X_train, y_train, X_test, y_test, encoder_dict = return_datasets_b(
            df,
            categorical_features=CATEGORICAL_FEATURES,
            cols_to_train=COLS_TO_TRAIN_ON,
        )

    clf = train_gbm(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        outcome_var=outcome_var,
        multi=meta[outcome_var]["multi"],
        hyperparam_choice=HYPERPARAMETER_CHOICE,
    )

    if save:
        encoder_paths = {
            "dest_ks4": FINAL_ENCODERS_PATH_DEST,
            "mat_y11": FINAL_ENCODERS_PATH_MAT,
            "eng_y11": FINAL_ENCODERS_PATH_ENG,
        }
        os.makedirs(os.path.dirname(encoder_paths[outcome_var]), exist_ok=True)
        with open(encoder_paths[outcome_var], "wb") as f:
            pickle.dump(encoder_dict, f)

        os.makedirs(os.path.dirname(meta[outcome_var]["save_path"]), exist_ok=True)
        clf.save_model(meta[outcome_var]["save_path"])
