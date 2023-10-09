import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from uni_asp.constants import DATAFRAME_PATH, SANITY_CHECKS_PATH
from uni_asp.modelling.model_pre import label_binary, label_multi


def schoolwise_comparisons_sixthform_enrollment_rate(df, df_true):
    """Different School-wise Stats and Comparisons of Students Enrolled or not-Enrolled in Sixth Form"""
    # Creating grouped dataframe for modelled test set
    grouped = df.groupby(["school", "label_dest_ks4"]).size().unstack(fill_value=0)
    grouped["total"] = grouped.sum(axis=1)
    grouped["Proportion of 0"] = grouped[0] / grouped["total"]
    grouped["Proportion of 1"] = grouped[1] / grouped["total"]

    # Creating grouped dataframe for true test set
    grouped_true = df_true.groupby(["school", "dest_ks4"]).size().unstack(fill_value=0)
    grouped_true["total"] = grouped_true.sum(axis=1)
    grouped_true["Proportion of 0"] = grouped_true[0] / grouped_true["total"]
    grouped_true["Proportion of 1"] = grouped_true[1] / grouped_true["total"]

    grouped["nature"] = "Classified"
    grouped_true["nature"] = "True"
    combined_grouped = pd.concat([grouped, grouped_true]).reset_index()

    # plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=grouped.index,
        y="Proportion of 0",
        data=grouped,
        color="pink",
        label="Not Sixth Form",
    )
    ax = sns.barplot(
        x=grouped.index,
        y="Proportion of 1",
        data=grouped,
        color="lightgreen",
        label="Sixth Form",
        bottom=grouped["Proportion of 0"],
    )
    ax.set_xlabel("School")
    ax.set_ylabel("Proportion")
    ax.set_title(
        "Modelled Proportions of Students Classified as Enrolling in Sixth Form by School"
    )
    ax.legend(title="Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Modelled Proportions of Students Classified as Enrolling in Sixth Form by School.png"
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=grouped_true.index,
        y="Proportion of 0",
        data=grouped_true,
        color="pink",
        label="Not Sixth Form",
    )
    ax = sns.barplot(
        x=grouped_true.index,
        y="Proportion of 1",
        data=grouped_true,
        color="lightgreen",
        label="Sixth Form",
        bottom=grouped_true["Proportion of 0"],
    )
    ax.set_xlabel("School")
    ax.set_ylabel("Proportion")
    ax.set_title(
        "True Proportions of Students Classified as Enrolling in Sixth Form by School"
    )
    ax.legend(title="Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/True Proportions of Students Classified as Enrolling in Sixth Form by School.png"
    )

    plt.figure(figsize=(14, 8))
    # Plot the combined bar plots with hue for differentiation
    ax = sns.barplot(
        x="school",
        y="Proportion of 1",
        hue="nature",
        data=combined_grouped,
        palette=["pink", "lightgreen"],
    )
    ax.set_xlabel("School")
    ax.set_ylabel("Proportion")
    ax.set_title(
        "Comparison of Classified and True Proportions of Students Enrolling in Sixth Form by School"
    )
    ax.legend(title="Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Classified and True Proportions of Students Enrolling in Sixth Form by School.png"
    )

    plt.figure(figsize=(14, 8))
    # Plot the combined bar plots with hue for differentiation
    ax = sns.barplot(
        x="school",
        y="Proportion of 0",
        hue="nature",
        data=combined_grouped,
        palette=["pink", "lightgreen"],
    )
    ax.set_xlabel("School")
    ax.set_ylabel("Proportion")
    ax.set_title(
        "Comparison of Classified and True Proportions of Students Not Enrolling in Sixth Form by School"
    )
    ax.legend(title="Class")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Classified and True Proportions of Students Not Enrolling in Sixth Form by School.png"
    )

    # Calculate the difference in proportions
    combined_grouped["Difference_1"] = abs(
        combined_grouped[combined_grouped["nature"] == "True"][
            "Proportion of 1"
        ].reset_index(drop=True)
        - combined_grouped[combined_grouped["nature"] == "Classified"][
            "Proportion of 1"
        ].reset_index(drop=True)
    )

    plt.figure(figsize=(14, 8))
    # Plot the bar plot showing the difference in proportions
    ax = sns.barplot(
        x="school", y="Difference_1", data=combined_grouped, palette="coolwarm"
    )
    ax.axhline(0, color="black", linewidth=0.8)  # Add a reference line at y=0
    ax.set_xlabel("School")
    ax.set_ylabel("Difference in Proportion")
    ax.set_title(
        "Difference in Proportions of Students Enrolling in Sixth Form (Classified vs. True) by School"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Difference in Proportions of Students Enrolling in Sixth Form (Classified vs. True) by School.png"
    )


def assign_at_risk_cols(df, col, preds, col_thresh, pred_thresh):
    df = df.copy()
    df["preds"] = preds
    col_name = col + "_" + "risk"
    df[col_name] = 0
    df.loc[
        np.logical_and(
            df[col].astype(float) > col_thresh, df["preds"].astype(float) <= pred_thresh
        ),
        col_name,
    ] = 1
    return df


def risk_score_df(df, df_true):
    # Classified Test Data
    df2 = df.copy()
    df_risk = assign_at_risk_cols(
        df2, "mat_y7", df2["prob_dest_ks4"], col_thresh=70, pred_thresh=0.4
    )
    df_risk = assign_at_risk_cols(
        df_risk, "mat_y8", df2["prob_dest_ks4"], col_thresh=80, pred_thresh=0.4
    )
    df_risk = assign_at_risk_cols(
        df_risk, "eng_y7", df2["prob_dest_ks4"], col_thresh=40, pred_thresh=0.5
    )
    df_risk = assign_at_risk_cols(
        df_risk, "eng_y8", df2["prob_dest_ks4"], col_thresh=50, pred_thresh=0.5
    )
    df_risk["risk_score"] = (
        df_risk["mat_y7_risk"]
        + df_risk["mat_y8_risk"]
        + df_risk["eng_y7_risk"]
        + df_risk["eng_y8_risk"]
    )

    # True Test Data
    df2_true = df_true.copy()
    df_risk_true = assign_at_risk_cols(
        df2_true, "mat_y7", df2_true["dest_ks4"], col_thresh=70, pred_thresh=0
    )
    df_risk_true = assign_at_risk_cols(
        df_risk_true, "mat_y8", df2_true["dest_ks4"], col_thresh=80, pred_thresh=0
    )
    df_risk_true = assign_at_risk_cols(
        df_risk_true, "eng_y7", df2_true["dest_ks4"], col_thresh=40, pred_thresh=0
    )
    df_risk_true = assign_at_risk_cols(
        df_risk_true, "eng_y8", df2_true["dest_ks4"], col_thresh=50, pred_thresh=0
    )
    df_risk_true["risk_score"] = (
        df_risk_true["mat_y7_risk"]
        + df_risk_true["mat_y8_risk"]
        + df_risk_true["eng_y7_risk"]
        + df_risk_true["eng_y8_risk"]
    )

    return df_risk, df_risk_true


def make_risk_plots(df_risk, df_risk_true):
    plt.figure(figsize=(8, 6))
    df_risk = df_risk.copy()
    df_risk_true = df_risk_true.copy()
    df_risk["Data"] = "Classified Data"
    df_risk_true["Data"] = "True Data"
    combined_df = pd.concat([df_risk, df_risk_true])

    # Plot the combined count plotfor comparison
    sns.countplot(x="risk_score", hue="Data", data=combined_df)
    plt.xlabel("Risk")
    plt.title("Comparison of Risk Scores Distribution for Students at-risk")
    plt.legend(title="Nature")
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Risk Scores Distribution for Students at-risk.png"
    )

    # True Test Data
    plt.figure(figsize=(8, 6))
    sns.countplot(x=df_risk_true["risk_score"], hue=df_risk_true["dest_ks4"])
    plt.xlabel("Risk")
    plt.title("Destination Analysis of Students at Risk (True Data)")
    plt.savefig(
        SANITY_CHECKS_PATH + "/Destination Analysis of Students at Risk (True Data).png"
    )


def maths_score_distribution_by_pred_dest(df_uni, df_not_uni):
    combined_df = pd.concat([df_uni, df_not_uni])

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(
        x="mat_y7", hue="label_dest_ks4", data=combined_df, element="step", bins=24
    )
    plt.xlabel("Year 7 Math Scores")
    plt.title(
        "Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (Classified)"
    )
    plt.legend(["Non-Sixth Form Students", "Sixth Form Students"])
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (Classified).png"
    )

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(
        x="mat_y8", hue="label_dest_ks4", data=combined_df, element="step", bins=24
    )
    plt.xlabel("Year 8 Math Scores")
    plt.title(
        "Comparison of Year 8 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (Classified)"
    )
    plt.legend(["Non-Sixth Form Students", "Sixth Form Students"])
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 8 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (Classified).png"
    )


def maths_score_distribution_by_true_dest(dft_uni, dft_not_uni):
    combined_dft = pd.concat([dft_uni, dft_not_uni])

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(
        x="mat_y7",
        hue="dest_ks4",
        data=combined_dft,
        element="step",
        bins=24,
        stat="percent",
    )
    plt.xlabel("Year 7 Math Scores")
    plt.title(
        "Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (True)"
    )
    plt.legend(["Non-Sixth Form Students", "Sixth Form Students"])
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (True).png"
    )

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(x="mat_y8", hue="dest_ks4", data=combined_dft, element="step", bins=24)
    plt.xlabel("Year 8 Math Scores")
    plt.title(
        "Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (True)"
    )
    plt.legend(["Non-Sixth Form Students", "Sixth Form Students"])
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form vs Not Enrolled in Sixth Form (True).png"
    )


def maths_score_distribution_comaprison_for_true_vs_pred_sixthform(df_uni, dft_uni):
    """Compares maths scores distributions for true sixth form enrollments vs predicted sixth form enrollments"""
    df_uni["nature"] = "Classified"
    dft_uni["nature"] = "True"
    combined_uni = pd.concat([df_uni, dft_uni])
    combined_uni = combined_uni.reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(x="mat_y7", hue="nature", data=combined_uni, element="step", bins=24)
    plt.xlabel("Year 7 Math Scores")
    plt.title(
        "Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form (Classified vs True)"
    )
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 7 Math Scores Distribution for Students Enrolled in Sixth Form (Classified vs True).png"
    )

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(x="mat_y8", hue="nature", data=combined_uni, element="step", bins=24)
    plt.xlabel("Year 8 Math Scores")
    plt.title(
        "Comparison of Year 8 Math Scores Distribution for Students Enrolled in Sixth Form (Classified vs True)"
    )
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 8 Math Scores Distribution for Students Enrolled in Sixth Form (Classified vs True).png"
    )


def maths_score_distribution_comaprison_for_true_vs_pred_notsixthform(
    df_not_uni, dft_not_uni
):
    """Compares maths scores distributions for true not-sixth form enrollments vs predicted not-sixth form enrollments"""
    df_not_uni["nature"] = "Classified"
    dft_not_uni["nature"] = "True"
    combined_not_uni = pd.concat([df_not_uni, dft_not_uni])
    combined_not_uni = combined_not_uni.reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(
        x="mat_y8", hue="nature", data=combined_not_uni, element="step", bins=24
    )
    plt.xlabel("Year 8 Math Scores")
    plt.title(
        "Comparison of Year 8 Math Scores Distribution for Students Not Enrolled in Sixth Form (Classified vs True)"
    )
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 8 Math Scores Distribution for Students Not Enrolled in Sixth Form (Classified vs True).png"
    )

    plt.figure(figsize=(10, 8))
    # Plot the combined histogram with hue for differentiation
    sns.histplot(
        x="mat_y7", hue="nature", data=combined_not_uni, element="step", bins=24
    )
    plt.xlabel("Year 7 Math Scores")
    plt.title(
        "Comparison of Year 7 Math Scores Distribution for Students Not Enrolled in Sixth Form (Classified vs True)"
    )
    plt.savefig(
        SANITY_CHECKS_PATH
        + "/Comparison of Year 7 Math Scores Distribution for Students Not Enrolled in Sixth Form (Classified vs True).png"
    )


def model_plots(df3):
    """Main function calling all other functions above to give respective plots"""
    df_uni = df3[df3["label_dest_ks4"] == 0]
    df_not_uni = df3[df3["label_dest_ks4"] == 1]

    df_true = df3.copy()
    dft_uni = df_true[df_true["dest_ks4"] == 0]
    dft_not_uni = df_true[df_true["dest_ks4"] == 1]

    # school-wise differences in classification percentages
    print(
        "School-wise Differences in Classification Percentages/Distribution of Enrolling in Sixth Form or Not:"
    )
    schoolwise_comparisons_sixthform_enrollment_rate(df3, df_true)

    # Risk Analysis Plots
    df_risk, df_risk_true = risk_score_df(df3, df_true)
    print("Risk Plots:")
    make_risk_plots(df_risk, df_risk_true)

    # Distribution plot (sixth vs not sixth) classified data:
    print("Year 7 and Year 8 Sixth Form vs Not Sixth Form Comparison (Class):")
    maths_score_distribution_by_pred_dest(df_uni, df_not_uni)

    # Distribution plot (sixth vs not sixth) true data:
    print("Year 7 and Year 8 Sixth Form vs Not Sixth Form Comparison (True):")
    maths_score_distribution_by_true_dest(dft_uni, dft_not_uni)

    # Distribution plots (sixth)
    print("Year 7 and Year 8 Sixth Form Comparison (Class vs True):")
    maths_score_distribution_comaprison_for_true_vs_pred_sixthform(df_uni, dft_uni)

    # Distribution plots (not sixth)
    print("Year 7 and Year 8 Not Sixth Form Comparison (Class vs True):")
    maths_score_distribution_comaprison_for_true_vs_pred_notsixthform(
        df_not_uni, dft_not_uni
    )


def proportion_analysis_of_at_risk_students(df, df2, df_mat, df_eng):
    df = df.copy()
    df = df[["upn", "dest_ks4", "eng_y11", "mat_y11"]]
    df2 = df2.copy()
    df_mat = df_mat.copy()
    df_eng = df_eng.copy()
    df_mat = df_mat[["upn", "label_gcse"]]
    df_eng = df_mat[["upn", "label_gcse"]]
    eng_mat = pd.merge(df_eng, df_mat, on="upn", how="inner", suffixes=["_eng", "_mat"])
    all_preds_merge = pd.merge(df2, eng_mat, on="upn", how="inner")
    pred_actual = all_preds_merge.merge(df, on="upn", how="left")
    df_checks = pred_actual.copy()
    df_checks = label_multi(df_checks, "eng_y11")
    df_checks = label_multi(df_checks, "mat_y11")
    # Sum columns to get risk scores
    df_checks["risk_true"] = df_checks[["mat_y11", "eng_y11"]].sum(axis=1)
    df_checks["risk_model"] = df_checks[["label_gcse_mat", "label_gcse_eng"]].sum(
        axis=1
    )
    actual_df = df_checks[df_checks["dest_ks4"] == 0]
    pred_df = df_checks[df_checks["labels"] == 0]
    at_risk_actual_any = actual_df[(actual_df["risk_true"] >= 1)]
    at_risk_pred_any = pred_df[(pred_df["risk_model"] >= 1)]
    # checking for common upns between two
    common_upn = set(at_risk_pred_any["upn"]).intersection(at_risk_actual_any["upn"])
    common_upn_list = list(common_upn)
    no_comm_upns = len(common_upn_list)
    prop = no_comm_upns / at_risk_actual_any.shape[0]
    with open(
        f"{SANITY_CHECKS_PATH}/proportion_of_correctly_identified_at_risk_students.txt",
        "w",
    ) as file:
        file.write(f"Total Number of Students at Risk: {at_risk_actual_any.shape[0]}")
        file.write(
            f"Number of Students of these identified to be at risk: {no_comm_upns}"
        )
        file.write(f"Proportion of correctly identified students to be at risk: {prop}")


def run_sanity_checks():
    os.makedirs(SANITY_CHECKS_PATH, exist_ok=True)
    df = pd.read_parquet(DATAFRAME_PATH)
    df = df.dropna(subset=["dest_ks4"])
    df["dest_ks4"] = df["dest_ks4"].apply(label_binary)

    df2 = pd.read_parquet(COHORT_4_5_RESULTS + "/model_results_dest_ks4.parquet")
    df2 = df2[["prob_dest_ks4", "label_dest_ks4", "upn"]]

    df3 = df2.merge(df, on=["upn"], how="inner")
    df_mat = pd.read_parquet(f"{COHORT_4_5_RESULTS}/model_results_mat_gcse.parquet")
    df_eng = pd.read_parquet(f"{COHORT_4_5_RESULTS}/model_results_eng_gcse.parquet")

    model_plots(df3)
