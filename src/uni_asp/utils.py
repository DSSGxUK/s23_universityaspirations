def reorder_columns(df, *, initial_cols):
    """Reorder the columns of a dataframe"""
    remaining_cols = [c for c in df.columns if c not in initial_cols]
    column_order = initial_cols + remaining_cols
    return df[column_order]
