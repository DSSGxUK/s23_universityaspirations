import pandas as pd


def format_gender(x):
    """
    Formats the 'gender' column in the pupil characteristics dataframe

    Args:
        x: A value from n a pandas series

    Returns:
        A formated value from a pandas series

    """

    if isinstance(x, str):
        return x
    elif pd.isnull(x):
        return "unknown"
    else:
        raise TypeError(f"Unexpected type: {type(x)} ({x=})")
