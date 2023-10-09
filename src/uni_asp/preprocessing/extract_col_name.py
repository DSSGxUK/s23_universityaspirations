import re


def extract_col_name(col_names, pattern):
    """Extract a unique column name based on a regular expression

    Args:
        col_names: a list of column names from a dataframe
        pattern: a regex pattern to extract a unique column name from col_names

    Returns:
        unique_name: a unique column name from col_names extrated using the given pattern

    """

    [unique_name] = [x for x in col_names if re.search(pattern, x, flags=re.IGNORECASE)]
    return unique_name
