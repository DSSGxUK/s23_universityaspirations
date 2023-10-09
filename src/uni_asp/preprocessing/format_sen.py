import pandas as pd


def format_sen(x):
    """
    Formats the 'sen' column in the pupil characteristics dataframe

    Args:
        x: A value from a pandas series

    Returns:
        A formatted value from a pandas series
    """

    if isinstance(x, str):
        if x == "No Special Educational Need":
            return False
        else:
            return True
    elif pd.isnull(x):
        return False
    else:
        raise TypeError(f"Unexpected type: {type(x)} ({x=})")
