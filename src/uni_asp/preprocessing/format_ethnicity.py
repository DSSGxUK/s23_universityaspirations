import numpy as np
import pandas as pd


def format_ethnicity(x):
    """formats the 'ethnicity' column in the pupil characteristics dataframe

    Args:
        df: A value from n a pandas series

    Returns:
        A formated value from a pandas series
    """

    options = {
        True: [
            "White Eastern European",
            "Any Other White Background",
            "White - British",
            "White Other",
            "White - English",
            "White European",
            "Portuguese",
            "White - Irish",
            "Other White British",
            "Traveller of Irish Heritage",
            "Italian",
            "White - Welsh",
            "White - Scottish",
            "White - Northern Irish",
            "White - Cornish",
        ],
        False: [
            "White and Asian",
            "White and Black Caribbean",
            "White and Black African",
            "Any Other Mixed Background",
            "Other Mixed Background",
            "White and Any Other Asian Background",
            "Black and Any Other Ethnic Group",
            "White and Chinese",
            "Chinese and Any Other Ethnic Group",
            "White and Any Other Ethnic Group",
            "African Asian",
            "White and Pakistani",
            "Asian and Black",
            "White and Indian",
            "Black and Chinese",
            "Asian and Any Other Ethnic Group",
            # white
            "Black Caribbean",
            "Black - Ghanaian",
            "Other Black African",
            "Black - African",
            "Any Other Black Background",
            "Black - Somali",
            "Black - Nigerian",
            "Other Black",
            "Black - Congolese",
            "Black - Sudanese",
            "Black - Angolan",
            "Black European",
            "Black - Sierra Leonean",
            "Black North American",
            # asian
            "Chinese",
            "Vietnamese",
            "Hong Kong Chinese",
            "Filipino",
            "Asian and Chinese",
            "Thai",
            "Japanese",
            "Other Chinese",
            "Malaysian Chinese",
            "Malay",
            "Korean",
            "Pakistani",
            "Bangladeshi",
            "Indian",
            "Other Pakistani",
            "Mirpuri Pakistani",
            "Kashmiri Pakistani",
            "Sri Lankan Tamil",
            "Sri Lankan Other",
            "Sri Lankan Sinhalese",
            "Nepali",
            "Any Other Asian Background",
            "Other Asian",
            "Afghan",
            "Polynesian",
            # middle_eastern
            "Turkish Cypriot",
            "Turkish",
            "Iraqi",
            "Iranian",
            "Arab Other",
            "Egyptian",
            "Lebanese",
            "Moroccan",
            "Yemeni",
            "Kurdish",
            "Turkish/ Turkish Cypriot",
            "Libyan",
            # eastern_european
            "Kosovan",
            "White Western European",
            "Albanian",
            "Serbian",
            "Greek/ Greek Cypriot",
            "Greek",
            "Greek Cypriot",
            "Croatian",
            "Bosnian- Herzegovinian",
            # roma
            "Other Gypsy/Roma",
            "Gypsy / Roma",
            "Roma",
            "Gypsy",
            # latino
            "Latin/ South/ Central American",
            # other
            "Other Ethnic Group",
            "Any Other Ethnic Group",
            "Information Not Yet Obtained",
            "Refused",
        ],
    }

    if pd.isnull(x):
        return np.nan

    elif isinstance(x, str):
        [x] = [keys for keys, vals in options.items() if x in vals]
        return x

    else:
        raise TypeError(f"Unexpected type: {type(x)} ({x=})")
