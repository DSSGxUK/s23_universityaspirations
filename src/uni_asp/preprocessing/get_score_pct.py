import pandas as pd

from uni_asp.preprocessing.cohort import assign_cohorts


def get_score_pct(df, max_scores_df):
    """
    Creates percentages for all KS3 scores based on maximum scores.

    Args:
        df: The formated pandas dataframe with original KS3 grades in all columns 'mat_y7', 'mat_y8', 'mat_y9',
        'eng_y7', 'eng_y8', and 'eng_y9'.

        max_score_df: A pandas dataframe with information about the max possible score in english and math for
        every year and year group. Note, it should be confirmed that the latest max scores are updated.

    Returns:
        A pandas dataframe with all ks3 scores converted to percentages in an additional column labelled based
        on the score column name and an "_pct" at the end.
    """
    max_scores_df = max_scores_df.copy()

    # Run cohorts function
    max_scores = assign_cohorts(max_scores_df, upn=False)

    # Pivot max_scores dataframe to merge with KS3 scores as a new max score column
    pivot_mat = (
        max_scores.pivot(index=["cohort"], columns=["year_group"], values="mat")
        .reset_index()
        .rename(columns={7: "mat_y7_max", 8: "mat_y8_max", 9: "mat_y9_max"})
    )
    pivot_eng = (
        max_scores.pivot(index=["cohort"], columns=["year_group"], values="eng")
        .reset_index()
        .rename(columns={7: "eng_y7_max", 8: "eng_y8_max", 9: "eng_y9_max"})
    )
    pivot_max_scores = pd.merge(pivot_mat, pivot_eng, on="cohort")

    # Merge max_scores pivoted dataframe with dataframe
    df_pct = pd.merge(
        df, pivot_max_scores, left_on="cohort", right_on="cohort", how="left"
    )

    # Calculate percentages
    df_pct["mat_y7_pct"] = round(df_pct["mat_y7"] / df_pct["mat_y7_max"] * 100)
    df_pct["mat_y8_pct"] = round(df_pct["mat_y8"] / df_pct["mat_y8_max"] * 100)
    df_pct["mat_y9_pct"] = round(df_pct["mat_y9"] / df_pct["mat_y9_max"] * 100)

    df_pct["eng_y7_pct"] = round(df_pct["eng_y7"] / df_pct["eng_y7_max"] * 100)
    df_pct["eng_y8_pct"] = round(df_pct["eng_y8"] / df_pct["eng_y8_max"] * 100)
    df_pct["eng_y9_pct"] = round(df_pct["eng_y9"] / df_pct["eng_y9_max"] * 100)

    cols = df_pct.filter(regex="_max").columns
    df_pct.drop(columns=cols, inplace=True)

    return df_pct
