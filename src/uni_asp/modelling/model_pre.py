import numpy as np
import pandas as pd
import sklearn
from scipy.stats import qmc
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from uni_asp.constants import (
    BETA,
    COHORTS_TO_TRAIN,
    HYPERPARAMETERS_LIST,
    INT_HYPERPARAMS,
    LABELS_SIXTH_FORM,
)


def encode_categorical(df, categorical_features):
    """
    Does Ordinal encoding on the categorical columns provided

    Args:
        df: dataframe to do ordinal encoding on
        categorical_features: list of features

    Returns:
        Tuple:
            df -> dataframe with encoding
            encoder_dict -> dictionary with encoders to do inverse transform later
    """
    encoder_dict = {}
    df = df.copy()
    for col in categorical_features:
        enc = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        df[col] = enc.fit_transform(df[col].values.reshape(-1, 1).tolist())
        encoder_dict[col] = enc
    return df, encoder_dict


def label_binary(x):
    """
    Encodes the ks4 destination as 1/0 depending on the list in the constants.py file
    where 1 = going to sixth form ; 0 = doesn't go to sixth form
    """
    if x in LABELS_SIXTH_FORM:
        return 1
    else:
        return 0


def label_multi(df, outcome_var):
    """
    Encodes GCSE grades in 3 categories ->
        U-3 : Low/0
        4-6 : Medium/1
        7-9: High/2
    """
    df = df.copy()
    cond = [
        df[outcome_var] <= 3,
        (df[outcome_var] > 3) & (df[outcome_var] <= 6),
        df[outcome_var] > 6,
    ]
    vals = [0, 1, 2]
    df[outcome_var] = np.select(cond, vals)
    return df


def get_one_nan_df(df):
    """
    Returns a dataframe of pupils having exactly one missing value in the KS3 grades
    """
    grades = ["mat_y7", "mat_y8", "eng_y7", "eng_y8"]
    one_nan_df = df[df[grades].isna().sum(axis="columns") == 1]
    return one_nan_df


def return_datasets_b(df, categorical_features, cols_to_train):
    """
    Returns the train and test datasets for KS4 destination predictions

    Training data includes all pupils with at most one missing KS3 score, while the
    testing data only includes pupils with all four KS3 scores (y7&8 maths & english).

    Args:
        df: Dataframe to process (created by the preprocessing step of the main
            pipeline)
        categorical_features: The column names of categorical features (for the encoding
            of categorical features)
        cols_to_train: The features to use when making the predictions, plus the UPN
            column
    """
    df = df[df["cohort"].isin(COHORTS_TO_TRAIN)]
    df = df[cols_to_train + ["dest_ks4"]]
    df = df.rename(
        {
            "mat_y7_pct": "mat_y7",
            "mat_y8_pct": "mat_y8",
            "eng_y7_pct": "eng_y7",
            "eng_y8_pct": "eng_y8",
        },
        axis="columns",
    )
    df = df.dropna(subset=["dest_ks4"], axis=0)
    df["dest_ks4"] = df["dest_ks4"].apply(label_binary)
    df, encoder_dict = encode_categorical(df, categorical_features)
    df_dropped = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df_dropped.drop(["dest_ks4"], axis=1),
        df_dropped["dest_ks4"],
        test_size=0.2,
        random_state=42,
    )
    one_nan_df = get_one_nan_df(df)
    X_train_nan = pd.concat([X_train, one_nan_df.drop(["dest_ks4"], axis=1)])
    y_train_nan = pd.concat([y_train, one_nan_df["dest_ks4"]])
    needed_cols = X_train_nan.columns.tolist()
    needed_cols.remove("upn")
    X_train_nan[needed_cols] = X_train_nan[needed_cols].apply(
        pd.to_numeric, errors="coerce", axis=1
    )
    y_train_nan = y_train_nan.astype(float)
    return X_train_nan, y_train_nan, X_test, y_test, encoder_dict


def return_datasets_m(df, outcome_var, categorical_features, cols_to_train):
    """
    Returns the train and test datasets for GCSE maths and english predictions

    Training data includes all pupils with at most one missing KS3 score, while the
    testing data only includes pupils with all four KS3 scores (y7&8 maths & english).

    Args:
        df: Dataframe to process (created by the preprocessing step of the main
            pipeline)
        outcome_var: The name of the target column ('mat_gcse' or 'eng_gcse')
        categorical_features: The column names of categorical features (for the encoding
            of categorical features)
        cols_to_train: The features to use when making the predictions, plus the UPN
            column
    """
    df = df[df["cohort"].isin(COHORTS_TO_TRAIN)]
    df = df[cols_to_train + [outcome_var]]
    df = df.rename(
        {
            "mat_y7_pct": "mat_y7",
            "mat_y8_pct": "mat_y8",
            "eng_y7_pct": "eng_y7",
            "eng_y8_pct": "eng_y8",
        },
        axis="columns",
    )
    df = df.dropna(subset=[outcome_var], axis=0)
    df = label_multi(df, outcome_var)
    df, encoder_dict = encode_categorical(df, categorical_features)
    df_dropped = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df_dropped.drop(outcome_var, axis=1),
        df_dropped[outcome_var],
        test_size=0.2,
        random_state=42,
    )
    one_nan_df = get_one_nan_df(df)
    X_train_nan = pd.concat([X_train, one_nan_df.drop([outcome_var], axis=1)])
    y_train_nan = pd.concat([y_train, one_nan_df[outcome_var]])
    needed_cols = X_train_nan.columns.tolist()
    needed_cols.remove("upn")
    X_train_nan[needed_cols] = X_train_nan[needed_cols].apply(
        pd.to_numeric, errors="coerce", axis=1
    )
    y_train_nan = y_train_nan.astype(float)
    return X_train_nan, y_train_nan, X_test, y_test, encoder_dict


def get_best_threshold(clf, X_input, y_input):
    """
    Calculates the threshold based on maximum F beta score where beta = 0.5 i.e.
    Precision is valued double as compared to recall because we are more sensitive
    to False Negatives.
    Returns two values : best threshold value, maximum fbeta score
    """
    best_fb, best_thresh = 0, 0
    y_pred = clf.predict(X_input.drop("upn", axis=1))
    for threshold in np.arange(0, 1.001, 0.001):
        y_pred_labels = (y_pred > threshold).astype(int)
        fb = fbeta_score(y_input, y_pred_labels, beta=BETA)
        if fb > best_fb:
            best_fb = fb
            best_thresh = threshold
    return best_thresh, best_fb


def generate_hypercube(n_points):
    """
    Generate a Latin Hypercube, allowing for integer parameters.

    Each dimension corresponds to a key in HYPERPARAMETERS_LIST, in the order they
    appear in the dictionary. Integer parameters are handled by taking the floor.
    """

    sampler = qmc.LatinHypercube(d=len(HYPERPARAMETERS_LIST), seed=0)
    sample = sampler.random(n=n_points)
    l_bounds, u_bounds = [], []
    for val in HYPERPARAMETERS_LIST.values():
        l_bounds.append(val[0])
        u_bounds.append(val[1])
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    final_params = {}
    for num, arrays in enumerate(sample_scaled):
        final_params[num] = {}
        for name, val in zip(HYPERPARAMETERS_LIST.keys(), arrays):
            if name in INT_HYPERPARAMS:
                final_params[num].update({name: int(np.floor(val))})
            else:
                final_params[num].update({name: round(val, 3)})
    return final_params
