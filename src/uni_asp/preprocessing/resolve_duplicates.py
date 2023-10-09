def resolve_duplicates(df, dup_cols, res_col, new_col_name=None):
    """
    Resolves duplicates by picking the highest grade

    Args:
        df: a pandas dataframe
        dup_cols: list of column names to apply pandas.duplicated() to
        res_col: string of the column name to resolve duplicates with i.e. grade

    Returns:
        A formatted dataframe
    """

    df = df.copy()
    if new_col_name:
        df[new_col_name] = df.duplicated(dup_cols, keep=False)

    return (
        df.sort_values(res_col, na_position="first")
        .drop_duplicates(dup_cols, keep="last")
        .dropna(subset=dup_cols)
    )
