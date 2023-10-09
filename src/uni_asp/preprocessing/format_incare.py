import pandas as pd


def format_incare(x):
    """
    Formats the 'in_care' column in the pupil characteristics dataframe

    Args:
        x: A value from a pandas series

    Returns:
        A formated value from a pandas series
    """

    options = {
        True: [
            # In care
            "Other/In Care",
            "Fostered",
            "With Parents (under the supervision of social services)",
            "Children's Home",
            "Fostered and Other/In Care",
            "Children's Home and Fostered",
            # Adopted
            "Adopted from care inside England or Wales",
            "Adopted from care (Inactive)",
            "Adopted from care outside England or Wales",
            "Adopted from care (Inactive) and Other/In Care",
            "Adopted from care (Inactive) and Adopted from care inside England or Wales",
            # Left care
            "Left care through special guardianship order (SGO)",
            "Left care through child arrangement order (CAO)",
            "Left care through residence order (RO)",
            "Left care through child arrangement order (CAO) and Left care through special guardianship order (SGO)",
            "Left care through special guardianship order (SGO) and Left care through child arrangement order (CAO)",
            # Other
            "Left care through child arrangement order (CAO) and Other/In Care",
            "Left care through child arrangement order (CAO) and Other/In Care",
            "Left care through special guardianship order (SGO) and Other/In Care",
            "Left care through child arrangement order (CAO) and Adopted from care (Inactive)",
            "Other/In Care and Left care through child arrangement order (CAO)",
            "Other/In Care and Left care through special guardianship order (SGO)",
            "Other/In Care and Left care through residence order (RO)",
            "Other/In Care and With Parents (under the supervision of social services)",
            "Other/In Care and Adopted from care (Inactive)",
        ],
    }

    if pd.isnull(x):
        return False

    elif isinstance(x, str):
        [x] = [keys for keys, vals in options.items() if x in vals]
        return x

    else:
        raise TypeError(f"Unexpected type: {type(x)} ({x=})")
