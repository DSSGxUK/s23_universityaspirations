import warnings

import pandas as pd


def read_excel_csv(path, xlsx=True, sheet=0):
    """
    Reads a .xlsx or .csv files

    Args:
        path: a string with a path to a folder without the extensions
        xlsx: Bool indicating whether the file to be read has .xlsx extension. If set to False, it will read .csv
        sheet: integers or strings indicating the excel sheet to be read

    Returns:
        A dataframe
    """
    if xlsx:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Data Validation extension is not supported",
            )
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Print area cannot be set to Defined name",
            )
            df = pd.read_excel(f"{path}.xlsx", sheet_name=sheet)
    else:
        df = pd.read_csv(f"{path}.csv")
        df.to_parquet(f"{path}.parquet")

    return df
