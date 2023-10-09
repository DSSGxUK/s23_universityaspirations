import pandas as pd
import pandera as pa

yeargroup_repetition_schema = pa.DataFrameSchema(
    {
        "upn": pa.Column(str),
        "year_group": pa.Column(int),
        "year_end": pa.Column(int),
        "repeat": pa.Column(bool),
    }
)


@pa.check_output(yeargroup_repetition_schema)
def resolve_repeated_yeargroups(dataframe):
    """
    This function is used to find pupils who repeat a certain grade.
    The later year is used to determine the cohort. The earlier year row
    is dropped to ensure 1 row per upn.
    """
    dataframe = dataframe.copy()
    n_repeats = dataframe.groupby(["upn", "year_group"])["year_end"].transform(
        "nunique"
    )
    dataframe["repeat"] = n_repeats > 1

    return (
        dataframe.sort_values(["upn", "year_end"])
        .drop_duplicates(subset=["upn", "year_group"], keep="last")
        .reset_index(drop=True)
    )


cohort_schema = pa.DataFrameSchema(
    {
        "upn": pa.Column(str, required=False),
        "year_group": pa.Column(int),
        "year_end": pa.Column(int),
        "cohort": pa.Column(int),
    }
)


@pa.check_output(cohort_schema)
def assign_cohorts(df, upn=True):
    """
    A function to assign cohorts to students based on their year group
    and year end. If your dataframe does not have upns, set upn = False.
    """
    df = df.copy()
    df["cohort"] = df["year_end"] - df["year_group"] - 2006
    if upn:
        max_cohort = df.groupby("upn")["cohort"].max().reset_index()
        df_merged = pd.merge(df, max_cohort, on="upn", suffixes=("", "_max"))
        df_merged["cohort"] = df_merged["cohort_max"]
        df_merged.drop("cohort_max", axis=1, inplace=True)
        return df_merged
    else:
        return df


def assign_cohort_info(df):
    df = df.copy()
    df = resolve_repeated_yeargroups(df)
    df = assign_cohorts(df)
    return df
