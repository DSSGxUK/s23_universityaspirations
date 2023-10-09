def clean_upn(df):
    """Removes UPNs that are not 13 characters long

    Args:
        df: A pandas dataframe with a 'upn' column

    Returns:
        A formatted pandas dataframe
    """
    upn_len = df["upn"].str.len()
    new_df = df[upn_len == 13]

    return new_df
