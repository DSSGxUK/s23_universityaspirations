import logging
import os

import numpy as np
import pandas as pd

from uni_asp.analysis.get_inference_table import get_inference_table
from uni_asp.constants import ANALYSIS_DIR, DATAFRAME_PATH, FINAL_CSV_PATH
from uni_asp.utils import reorder_columns

# Create a logger. Using __name__ for the name of the logger is a standard pattern.
logger = logging.getLogger(__name__)


def risk_analysis():
    # Import model results
    dest = pd.read_parquet(
        os.path.join(FINAL_CSV_PATH, "model_results_dest_ks4.parquet")
    )
    eng = pd.read_parquet(
        os.path.join(FINAL_CSV_PATH, "model_results_eng_gcse.parquet")
    )
    mat = pd.read_parquet(
        os.path.join(FINAL_CSV_PATH, "model_results_mat_gcse.parquet")
    )
    # Mege destinations with english
    dest_eng = dest.merge(
        eng[["upn", "prob_low_gcse", "prob_med_gcse", "prob_high_gcse", "label_gcse"]],
        on="upn",
        how="inner",
        validate="one_to_one",
    )
    # Merge destinations with maths
    dest_mat = dest.merge(
        mat[["upn", "prob_low_gcse", "prob_med_gcse", "prob_high_gcse", "label_gcse"]],
        on="upn",
        how="inner",
        validate="one_to_one",
    )
    # Merge the two dataframe
    dest_eng_mat = dest_eng.merge(
        dest_mat[
            ["upn", "prob_low_gcse", "prob_med_gcse", "prob_high_gcse", "label_gcse"]
        ],
        on="upn",
        how="outer",
        validate="one_to_one",
        suffixes=["_eng", "_mat"],
    )
    # Merge dataframe with schools from full_df
    full_df = pd.read_parquet(DATAFRAME_PATH)
    dest_eng_mat = dest_eng_mat.merge(
        full_df[["upn", "school"]], on="upn", how="left", validate="one_to_one"
    )
    # Make a binary columns based on "gender"
    dest_eng_mat["is_male"] = dest_eng_mat["gender"].replace(
        {"Male": True, "Female": False}
    )
    # Create a column that combines English and Maths categories
    dest_eng_mat["comb_eng_mat"] = dest_eng_mat[
        ["label_gcse_eng", "label_gcse_mat"]
    ].sum(axis=1)
    # Create a column to reflect different at-risk levels
    at_risk_conditions = [
        dest_eng_mat["comb_eng_mat"] == 0,
        (dest_eng_mat["comb_eng_mat"] == 1) & (dest_eng_mat["label_dest_ks4"] == 0),
        (dest_eng_mat["comb_eng_mat"] == 2) & (dest_eng_mat["label_dest_ks4"] == 0),
        (dest_eng_mat["comb_eng_mat"] == 3) & (dest_eng_mat["label_dest_ks4"] == 0),
        (dest_eng_mat["comb_eng_mat"] == 4) & (dest_eng_mat["label_dest_ks4"] == 0),
        dest_eng_mat["label_dest_ks4"] == 1,
    ]
    at_risk_levels = ["undef", "risk1", "risk2", "risk3", "risk4", "no_risk"]
    dest_eng_mat["risk_level"] = np.select(at_risk_conditions, at_risk_levels)
    # Create a dataframe to include risk information at the UPN level
    risk_df = dest_eng_mat.copy()
    # Add column to carry risk analysis with 1 at-risk and 1 not-at-risk categories
    risk_df["at_risk"] = risk_df["risk_level"].replace(
        {
            "undef": np.nan,
            "risk1": True,
            "risk2": True,
            "risk3": True,
            "risk4": True,
            "no_risk": False,
        }
    )

    # Reorder columns to be friendly when saving
    risk_df = reorder_columns(
        risk_df,
        initial_cols=[
            "upn",
            "at_risk",
            "risk_level",
            "prob_dest_ks4",
            "label_dest_ks4",
            "threshold",
            "prob_low_gcse_eng",
            "prob_med_gcse_eng",
            "prob_high_gcse_eng",
            "label_gcse_eng",
            "prob_low_gcse_mat",
            "prob_med_gcse_mat",
            "prob_high_gcse_mat",
            "label_gcse_mat",
            "comb_eng_mat",
        ],
    )

    # Save the at-risk dataframe
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)
    file_path = os.path.join(ANALYSIS_DIR, "risk_analysis")
    logger.info(f"Saving file to {file_path}.parquet")
    risk_df.to_parquet(f"{file_path}.parquet")
    risk_df.to_csv(f"{file_path}.csv", index=False)

    # Create a dataframe to carry out risk analysis with 4 at-risk and 1 not-at-risk categories
    risk_df_for_analysis = risk_df.loc[risk_df["risk_level"] != "undef"].copy()

    # Define variables to group UPNS for risk-analysis
    groups = [
        # Geographical aggregations
        "school",
        "local_auth",
        "cluster",
        "north_south",
        "united_learning",
        # School characteristics aggregations
        "type",
        "phase",
        "ofstedrating",
    ]
    # Define features to perform risk analysis for
    features = [
        "eal",
        "sen",
        "premium",
        "is_white",
        "in_care",
        "is_male",
        "eng_y7_pct",
        "mat_y7_pct",
        "eng_y8_pct",
        "mat_y8_pct",
    ]
    # Perform risk analysis for each group and feature
    logger.info("Performing risk analysis...")
    for g in groups:
        logger.info(f"   at the {g} level")
        for f in features:
            # Carry out risk analysis with 2 categories (1 at-risk, 1 not at-risk)
            get_inference_table(risk_df_for_analysis, "at_risk", feature=f, agg_level=g)
            # Cary out risk analysis with 5 categories (4 at-risk, 1 not at-risk)
            get_inference_table(
                risk_df_for_analysis, "risk_level", feature=f, agg_level=g
            )
