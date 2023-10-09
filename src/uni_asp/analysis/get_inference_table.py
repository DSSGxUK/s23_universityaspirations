import os

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import binomtest

from uni_asp.constants import ANALYSIS_DIR


def get_inference_table(df, risk_column, feature, agg_level):
    """
    Performs risk-analysis for a particular feature and grouping variable and saves the results as a .csv

    Args:
        df: A dataframe to with one UPN per row to perform risk analysis on
        risk_column:  A string that is a column in df and which classifies UPNs across different risk groups
        feature: A string that is a column in df to perform risk analysis for
        agg_level: A string that is a column in df to group UPNs and perform the risk analysis at a particular level of aggreagtion

    Returns:
        A dataframe to analyze the difference between at-risk and not-at-risk groups given feature and an aggregation level
    """

    # Create an empty dataframe to store results
    inference_df = pd.DataFrame()

    # For Binary variables
    if feature in ["eal", "sen", "premium", "in_care", "is_white", "is_male"]:
        # Compute counts and confidence intervals for each risk level
        for risk_lvl in df[risk_column].unique().tolist():
            subset_df = df[df[risk_column] == risk_lvl]
            # Obtain inference at the United Learning level
            if agg_level == "united_learning":
                current_df = pd.DataFrame()
                # Compute the total number of students at risk ath the UL level
                current_df["upn_count"] = subset_df["upn"].count()
                # Compute the total number of students at risk ath the UL level with a True for this feature
                current_df["feature_count"] = subset_df[feature].sum()
                current_df["group"] = agg_level
            else:
                # Compute the total number of students at risk per cluster
                total_count = (
                    subset_df[[agg_level, "upn"]]
                    .groupby(agg_level)
                    .count()
                    .reset_index()
                )
                # Compute the total number of students at risk per cluster with a True for this feature
                feature_count = (
                    subset_df[[agg_level, feature]]
                    .groupby(agg_level)
                    .sum()
                    .reset_index()
                )
                # Create a dataframe with these two series
                current_df = total_count.merge(
                    feature_count, on=agg_level, how="inner", validate="one_to_one"
                )
                current_df = current_df.rename(
                    columns={
                        agg_level: "group",
                        "upn": "upn_count",
                        feature: "feature_count",
                    }
                )
            current_df[risk_column] = risk_lvl
            # Compute the lower and upper confidence intervals for the probability of having True for this feature conditional on cluster and risk level
            lower_ci = []
            upper_ci = []
            for g in current_df["group"].unique().tolist():
                ci = binomtest(
                    k=int(
                        current_df["feature_count"][current_df["group"] == g].values[0]
                    ),
                    n=int(current_df["upn_count"][current_df["group"] == g].values[0]),
                    p=0.95,
                ).proportion_ci()
                lower_ci.append(ci[0])
                upper_ci.append(ci[1])
            # Add columns with conditional probability and confidence intervals
            current_df["feature_prob"] = (
                current_df["feature_count"] / current_df["upn_count"]
            )
            current_df["lower_ci"] = lower_ci
            current_df["upper_ci"] = upper_ci

            inference_df = pd.concat([inference_df, current_df])

    # For continuous variables
    elif feature in ["eng_y7_pct", "mat_y7_pct", "eng_y8_pct", "mat_y8_pct"]:
        # Compute counts and confidence intervals for each risk level
        for risk_lvl in df[risk_column].unique().tolist():
            subset_df = df[df[risk_column] == risk_lvl]
            # Obtain inference at the United Learning level
            if agg_level == "united_learning":
                current_df = pd.DataFrame()
                # Compute the total number of students at risk ath the UL level
                current_df["upn_count"] = subset_df["upn"].count()
                # Compute the total number of students at risk at the UL level with a True for this feature
                current_df["feature_count"] = subset_df[feature].mean()
                # Add a column about level of aggregation
                current_df["group"] = agg_level
            else:
                # Compute the total number of students at risk per cluster
                total_count = (
                    subset_df[[agg_level, "upn"]]
                    .groupby(agg_level)
                    .count()
                    .reset_index()
                )
                # Compute the total number of students at risk per cluster with a True for this feature
                feature_mean = (
                    subset_df[[agg_level, feature]]
                    .groupby(agg_level)
                    .mean()
                    .reset_index()
                )
                # Create a dataframe with these two series
                current_df = total_count.merge(
                    feature_mean, on=agg_level, how="inner", validate="one_to_one"
                )
                current_df = current_df.rename(
                    columns={
                        agg_level: "group",
                        "upn": "upn_count",
                        feature: "feature_mean",
                    }
                )
            current_df[risk_column] = risk_lvl
            # Compute the lower and upper confidence intervals for the mean grade conditional on cluster and risk level
            lower_ci = []
            upper_ci = []
            for g in current_df["group"].unique().tolist():
                # Create a list with all the grades for this feature in this group to compute standard error
                if agg_level == "united_learning":
                    values_list = subset_df[feature].values
                else:
                    values_list = subset_df[feature][subset_df[agg_level] == g].values
                # If sample size is larger than 30, return upper and lower confidence intervals
                if current_df["upn_count"][current_df["group"] == g].values[0] >= 30:
                    ci = st.norm.interval(
                        loc=current_df["feature_mean"][current_df["group"] == g].values[
                            0
                        ],
                        scale=st.sem(values_list),
                        confidence=0.95,
                    )
                    lower_ci.append(ci[0])
                    upper_ci.append(ci[1])
                # If sample size is smaller than 30, return NAs as upper and lower confidence intervals
                else:
                    lower_ci.append(np.nan)
                    upper_ci.append(np.nan)

            current_df["lower_ci"] = lower_ci
            current_df["upper_ci"] = upper_ci
            inference_df = pd.concat([inference_df, current_df])

    else:
        raise ValueError(f"{feature} is an unsupported feature")

    # Add feature and aggregation level columns
    inference_df["feature"] = feature
    inference_df["agg_level"] = agg_level

    # Save results
    if df[risk_column].nunique() == 2:
        path_to_save = os.path.join(ANALYSIS_DIR, "binary_analysis", agg_level)
    elif df[risk_column].nunique() == 5:
        path_to_save = os.path.join(ANALYSIS_DIR, "multi_category_analysis", agg_level)
    else:
        raise ValueError(
            f"Please specify a folder to save risk analysis with {df[risk_column].nunique()} categories"
        )

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    file_name = f"{feature}_{agg_level}_risk_analysis.csv"
    inference_df.to_csv(os.path.join(path_to_save, file_name), index=False)
