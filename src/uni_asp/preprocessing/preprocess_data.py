import logging
import os

import pandas as pd

from uni_asp.constants import PROCESSED_DIR, RAW_DIR
from uni_asp.preprocessing.cohort import assign_cohort_info
from uni_asp.preprocessing.format_columns import format_columns
from uni_asp.preprocessing.format_school import format_school
from uni_asp.preprocessing.get_score_pct import get_score_pct
from uni_asp.preprocessing.read_excel_csv import read_excel_csv
from uni_asp.preprocessing.resolve_duplicates import resolve_duplicates


logger = logging.getLogger(__name__)


def preprocess_data():
    ### Import datasets
    logger.info("Loading data...")

    # School-level information
    logger.info("    School-level information (2 dataframes)")
    school_names_raw = read_excel_csv(
        os.path.join(RAW_DIR, "school_level/school_name_list")
    )
    school_info_raw = read_excel_csv(
        os.path.join(RAW_DIR, "school_level/school_info"), xlsx=False
    )
    # Pupil characteristics
    logger.info("    Pupil characteristics (6 dataframes)")
    pupchar_16_raw = read_excel_csv(
        os.path.join(RAW_DIR, "pupil_char/pupchar_2015_16"), xlsx=False
    )
    pupchar_17_raw = read_excel_csv(
        os.path.join(RAW_DIR, "pupil_char/pupchar_2016_17"), xlsx=False
    )
    pupchar_18_raw = read_excel_csv(
        os.path.join(RAW_DIR, "pupil_char/pupchar_2017_18"), xlsx=False
    )
    pupchar_21_raw = read_excel_csv(os.path.join(RAW_DIR, "pupil_char/pupchar_2020_21"))
    pupchar_22_raw = read_excel_csv(os.path.join(RAW_DIR, "pupil_char/pupchar_2021_22"))
    pupchar_23_raw = read_excel_csv(os.path.join(RAW_DIR, "pupil_char/pupchar_2022_23"))
    # Destinations
    logger.info("    Destinations (5 dataframes)")
    dest_19_ks4_raw = read_excel_csv(
        os.path.join(RAW_DIR, "destinations/destinations_2019"), sheet="KS4 pupildata"
    )
    dest_19_ks5_raw = read_excel_csv(
        os.path.join(RAW_DIR, "destinations/destinations_2019"), sheet="KS5 pupildata"
    )
    dest_21_ks4_raw = read_excel_csv(
        os.path.join(RAW_DIR, "destinations/destinations_2021_Y11"), sheet="Data"
    )
    dest_21_ks5_raw = read_excel_csv(
        os.path.join(RAW_DIR, "destinations/destinations_2021_Y13"), sheet="Data"
    )
    dest_22_raw = read_excel_csv(
        os.path.join(RAW_DIR, "destinations/destinations_2022"), sheet="Data"
    )
    # Conduct
    logger.info("    Conduct (10 dataframes)")
    att_sus_exp_16_raw = read_excel_csv(
        os.path.join(RAW_DIR, "conduct/att_exp_2015_16"), xlsx=False
    )
    att_sus_exp_17_raw = read_excel_csv(
        os.path.join(RAW_DIR, "conduct/att_exp_2016_17"), xlsx=False
    )
    sus_exp_18_raw = read_excel_csv(
        os.path.join(RAW_DIR, "conduct/att_exp_2017_18"), xlsx=False
    )
    att_18_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/att_2017_18"))
    sus_21_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/sus_2020_21"))
    sus_22_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/sus_2021_22"))
    sus_23_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/sus_2022_23"))
    exp_21_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/exp_2020_21"))
    exp_22_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/exp_2021_22"))
    exp_23_raw = read_excel_csv(os.path.join(RAW_DIR, "conduct/exp_2022_23"))
    # Year 7
    logger.info("    Year 7 (8 dataframes)")
    eng_y7_17_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2017/eng_y7_2017"))
    mat_y7_17_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2017/mat_y7_2017"))
    eng_y7_19_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2019/eng_y7_2019"))
    mat_y7_19_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2019/mat_y7_2019"))
    eng_y7_21_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2021/eng_y7_2021"))
    mat_y7_21_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2021/mat_y7_2021"))
    eng_y7_22_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2022/eng_y7_2022"))
    mat_y7_22_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2022/mat_y7_2022"))
    # Year 8
    logger.info("    Year 8 (8 dataframes)")
    eng_y8_17_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2017/eng_y8_2017"))
    mat_y8_17_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2017/mat_y8_2017"))
    eng_y8_19_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2019/eng_y8_2019"))
    mat_y8_19_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2019/mat_y8_2019"))
    eng_y8_21_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2021/eng_y8_2021"))
    mat_y8_21_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2021/mat_y8_2021"))
    eng_y8_22_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2022/eng_y8_2022"))
    mat_y8_22_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2022/mat_y8_2022"))
    # Year 9
    logger.info("    Year 9 (9 dataframes)")
    eng_y9_17_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2017/eng_y9_2017"))
    mat_y9_17_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2017/mat_y9_2017"))
    eng_y9_19_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2019/eng_y9_2019"))
    mat_y9_19_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2019/mat_y9_2019"))
    eng_y9_21_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2021/eng_y9_2021"))
    mat_y9_21_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2021/mat_y9_2021"))
    eng_y9_22_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2022/eng_y9_2022"))
    mat_y9_22_raw = read_excel_csv(os.path.join(RAW_DIR, "KS3_EOY/2022/mat_y9_2022"))
    # Year 7 to 9 2018
    y7_to_y9_18_raw = read_excel_csv(
        os.path.join(RAW_DIR, "KS3_EOY/2018/ks3_2018"),
        sheet=["Y7_Eng", "Y8_Eng", "Y9_Eng", "Y7_Mat", "Y8_Mat", "Y9_Mat"],
    )
    # Year 10
    logger.info("    Year 10 (9 dataframes)")
    eng_y10_17_raw = read_excel_csv(os.path.join(RAW_DIR, "Y10_EOY/2017/eng_y10_2017"))
    mat_y10_17_raw = read_excel_csv(os.path.join(RAW_DIR, "Y10_EOY/2017/mat_y10_2017"))
    y10_18_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2018/y10_2018"), sheet="Year 10"
    )
    eng_y10_19_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2019/eng_y10_2019"), sheet="Data"
    )
    mat_y10_19_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2019/mat_y10_2019"),
        sheet=["Data FND", "Data HGH"],
    )
    y10_21_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2021/y10_2021"),
        sheet=["EngLit", "EngLang", "Maths_higher", "Maths_foundation"],
    )
    eng_lan_y10_22_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2022/eng_lan_y10_2022"), sheet="Yr10"
    )
    eng_lit_y10_22_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2022/eng_lit_y10_2022"), sheet="Yr10"
    )
    mat_y10_22_raw = read_excel_csv(
        os.path.join(RAW_DIR, "Y10_EOY/2022/mat_y10_2022"),
        sheet=["FOUNDATION", "HIGHER"],
    )
    # GCSEs
    logger.info("    GCSEs (5 dataframes)")
    gcse_17_raw = read_excel_csv(os.path.join(RAW_DIR, "GCSE_Alevels/gcse_2017"))
    gcse_18_raw = read_excel_csv(os.path.join(RAW_DIR, "GCSE_Alevels/gcse_2018"))
    gcse_19_raw = read_excel_csv(os.path.join(RAW_DIR, "GCSE_Alevels/gcse_2019"))
    gcse_21_raw = read_excel_csv(
        os.path.join(RAW_DIR, "GCSE_Alevels/gcse_alevels_2021"),
        sheet="Academies - KS4 - subjects",
    )
    gcse_22_raw = read_excel_csv(os.path.join(RAW_DIR, "GCSE_Alevels/gcse_2022"))

    # Max scores (for standardisation)
    max_scores = read_excel_csv(
        os.path.join(RAW_DIR, "KS3_EOY/max_score_map"), xlsx=False
    )

    ### Format columns
    logger.info("Formatting columns")

    # School-level information
    school_names = format_columns(school_names_raw, "school_names")
    school_info = format_columns(school_info_raw, "school_info")

    # Pupil characteristics
    pupchar_16 = format_columns(pupchar_16_raw, "pup_char", 2016)
    pupchar_17 = format_columns(pupchar_17_raw, "pup_char", 2017)
    pupchar_18 = format_columns(pupchar_18_raw, "pup_char", 2018)
    pupchar_21 = format_columns(pupchar_21_raw, "pup_char", 2021)
    pupchar_22 = format_columns(pupchar_22_raw, "pup_char", 2022)
    pupchar_23 = format_columns(pupchar_23_raw, "pup_char", 2023)
    # Destinations
    dest_19_ks4 = format_columns(dest_19_ks4_raw, "dest_type1", 2019, year_group="ks4")
    dest_19_ks5 = format_columns(dest_19_ks5_raw, "dest_type1", 2019, year_group="ks5")
    dest_21_ks4 = format_columns(dest_21_ks4_raw, "dest_type1", 2021, year_group="ks4")
    dest_21_ks5 = format_columns(dest_21_ks5_raw, "dest_type1", 2021, year_group="ks5")
    dest_22 = format_columns(dest_22_raw, "dest_type2", 2022)
    # Conduct
    att_sus_exp_16 = format_columns(att_sus_exp_16_raw, "att_exp_susp", 2016)
    att_sus_exp_17 = format_columns(att_sus_exp_17_raw, "att_exp_susp", 2017)
    sus_exp_18 = format_columns(sus_exp_18_raw, "att_exp_susp", 2018)
    att_18 = format_columns(att_18_raw, "att", 2018)
    sus_21 = format_columns(sus_21_raw, "susp", 2021)
    sus_22 = format_columns(sus_22_raw, "susp", 2022)
    sus_23 = format_columns(sus_23_raw, "susp", 2023)
    exp_21 = format_columns(exp_21_raw, "exp", 2021)
    exp_22 = format_columns(exp_22_raw, "exp", 2022)
    exp_23 = format_columns(exp_23_raw, "exp", 2023)
    # Year 7
    eng_y7_17 = format_columns(
        eng_y7_17_raw, "ks3_type1", 2017, year_group=7, subject="eng"
    )
    mat_y7_17 = format_columns(
        mat_y7_17_raw, "ks3_type1", 2017, year_group=7, subject="mat"
    )
    eng_y7_19 = format_columns(
        eng_y7_19_raw, "ks3_type1", 2019, year_group=7, subject="eng"
    )
    mat_y7_19 = format_columns(
        mat_y7_19_raw, "ks3_type1", 2019, year_group=7, subject="mat"
    )
    eng_y7_21 = format_columns(
        eng_y7_21_raw, "ks3_type1", 2021, year_group=7, subject="eng"
    )
    mat_y7_21 = format_columns(
        mat_y7_21_raw, "ks3_type1", 2021, year_group=7, subject="mat"
    )
    eng_y7_22 = format_columns(
        eng_y7_22_raw, "ks3_type1", 2022, year_group=7, subject="eng"
    )
    mat_y7_22 = format_columns(
        mat_y7_22_raw, "ks3_type1", 2022, year_group=7, subject="mat"
    )
    # Year 8
    eng_y8_17 = format_columns(
        eng_y8_17_raw, "ks3_type1", 2017, year_group=8, subject="eng"
    )
    mat_y8_17 = format_columns(
        mat_y8_17_raw, "ks3_type1", 2017, year_group=8, subject="mat"
    )
    eng_y8_19 = format_columns(
        eng_y8_19_raw, "ks3_type1", 2019, year_group=8, subject="eng"
    )
    mat_y8_19 = format_columns(
        mat_y8_19_raw, "ks3_type1", 2019, year_group=8, subject="mat"
    )
    eng_y8_21 = format_columns(
        eng_y8_21_raw, "ks3_type1", 2021, year_group=8, subject="eng"
    )
    mat_y8_21 = format_columns(
        mat_y8_21_raw, "ks3_type1", 2021, year_group=8, subject="mat"
    )
    eng_y8_22 = format_columns(
        eng_y8_22_raw, "ks3_type1", 2022, year_group=8, subject="eng"
    )
    mat_y8_22 = format_columns(
        mat_y8_22_raw, "ks3_type1", 2022, year_group=8, subject="mat"
    )
    # Year 9
    eng_y9_17 = format_columns(
        eng_y9_17_raw, "ks3_type1", 2017, year_group=9, subject="eng"
    )
    mat_y9_17 = format_columns(
        mat_y9_17_raw, "ks3_type1", 2017, year_group=9, subject="mat"
    )
    eng_y9_19 = format_columns(
        eng_y9_19_raw, "ks3_type1", 2019, year_group=9, subject="eng"
    )
    mat_y9_19 = format_columns(
        mat_y9_19_raw, "ks3_type1", 2019, year_group=9, subject="mat"
    )
    eng_y9_21 = format_columns(
        eng_y9_21_raw, "ks3_type1", 2021, year_group=9, subject="eng"
    )
    mat_y9_21 = format_columns(
        mat_y9_21_raw, "ks3_type1", 2021, year_group=9, subject="mat"
    )
    eng_y9_22 = format_columns(
        eng_y9_22_raw, "ks3_type1", 2022, year_group=9, subject="eng"
    )
    mat_y9_22 = format_columns(
        mat_y9_22_raw, "ks3_type1", 2022, year_group=9, subject="mat"
    )
    # Year 7 to 9 2018
    y7_to_y9_18 = format_columns(y7_to_y9_18_raw, "ks3_type2", 2018)
    # Year 10
    eng_y10_17 = format_columns(
        eng_y10_17_raw, "y10_2017", 2017, year_group=10, subject="eng"
    )
    mat_y10_17 = format_columns(
        mat_y10_17_raw, "y10_2017", 2017, year_group=10, subject="mat"
    )
    y10_18 = format_columns(y10_18_raw, "y10_2018", 2018, year_group=10)
    eng_y10_19 = format_columns(
        eng_y10_19_raw, "y10_2019", 2019, year_group=10, subject="eng"
    )
    mat_y10_19 = format_columns(
        mat_y10_19_raw, "y10_2019", 2019, year_group=10, subject="mat"
    )
    y10_21 = format_columns(y10_21_raw, "y10_2021", 2021, year_group=10)
    eng_lan_y10_22 = format_columns(
        eng_lan_y10_22_raw, "y10_2022", 2022, year_group=10, subject="eng_lan"
    )
    eng_lit_y10_22 = format_columns(
        eng_lit_y10_22_raw, "y10_2022", 2022, year_group=10, subject="eng_lit"
    )
    mat_y10_22 = format_columns(
        mat_y10_22_raw, "y10_2022", 2022, year_group=10, subject="mat"
    )
    # GCSEs
    gcse_17 = format_columns(gcse_17_raw, "gcse_type1", 2017, year_group=11)
    gcse_18 = format_columns(gcse_18_raw, "gcse_type1", 2018, year_group=11)
    gcse_19 = format_columns(gcse_19_raw, "gcse_type2", 2019, year_group=11)
    gcse_21 = format_columns(gcse_21_raw, "gcse_type3", 2021, year_group=11)
    gcse_22 = format_columns(gcse_22_raw, "gcse_type4", 2022, year_group=11)

    ### Consolidate datasets
    logger.info("Consolidating dataframes")

    # School level variables
    school_names_sch = school_names.sort_values(
        "cluster", na_position="first"
    ).drop_duplicates("school", keep="last")
    school_info = school_info.sort_values(
        ["ofstedrating", "admissions_pol"], na_position="first"
    ).drop_duplicates("school", keep="last")
    school_level = school_names_sch.merge(
        school_info, on=["school"], how="outer", validate="one_to_one"
    )

    # Key Stage 3
    # Year 7
    eng_y7 = pd.concat([eng_y7_17, eng_y7_19, eng_y7_21, eng_y7_22])
    eng_y7_nodupls = resolve_duplicates(
        eng_y7, ["upn", "year_end", "year_group", "school"], "eng_y7"
    )
    mat_y7 = pd.concat([mat_y7_17, mat_y7_19, mat_y7_21, mat_y7_22])
    mat_y7_nodupls = resolve_duplicates(
        mat_y7, ["upn", "year_end", "year_group", "school"], "mat_y7"
    )
    y7_non18 = eng_y7_nodupls.merge(
        mat_y7_nodupls,
        how="inner",
        on=["upn", "year_end", "year_group", "school"],
        validate="one_to_one",
    )
    y7_18 = resolve_duplicates(
        y7_to_y9_18[0],
        ["upn", "year_end", "year_group", "school"],
        ["mat_y7", "eng_y7"],
    )
    # Combine all Y7 data
    y7_raw = pd.concat([y7_non18, y7_18])
    # Resolve transfers
    y7_raw = resolve_duplicates(
        y7_raw, ["upn", "year_end", "year_group"], ["mat_y7", "eng_y7"], "transfer"
    )
    # Assign cohorts
    y7 = assign_cohort_info(y7_raw).drop(columns=["year_group"])

    # Year 8
    eng_y8 = pd.concat([eng_y8_17, eng_y8_19, eng_y8_21, eng_y8_22])
    eng_y8_nodupls = resolve_duplicates(
        eng_y8, ["upn", "year_end", "year_group", "school"], "eng_y8"
    )
    mat_y8 = pd.concat([mat_y8_17, mat_y8_19, mat_y8_21, mat_y8_22])
    mat_y8_nodupls = resolve_duplicates(
        mat_y8, ["upn", "year_end", "year_group", "school"], "mat_y8"
    )
    y8_non18 = eng_y8_nodupls.merge(
        mat_y8_nodupls,
        how="inner",
        on=["upn", "year_end", "year_group", "school"],
        validate="one_to_one",
    )
    y8_18 = resolve_duplicates(
        y7_to_y9_18[1],
        ["upn", "year_end", "year_group", "school"],
        ["mat_y8", "eng_y8"],
    )
    # Combine all Y8 data
    y8_raw = pd.concat([y8_non18, y8_18])
    # Resolve transfers
    y8_raw = resolve_duplicates(
        y8_raw, ["upn", "year_end", "year_group"], ["mat_y8", "eng_y8"], "transfer"
    )
    # Assign cohorts
    y8 = (
        assign_cohort_info(y8_raw)
        .drop(columns=["year_group"])
        .sort_values(["upn", "year_end"])
        .drop_duplicates(["upn"], keep="last")
    )

    # Year 9
    eng_y9 = pd.concat([eng_y9_17, eng_y9_19, eng_y9_21, eng_y9_22])
    eng_y9_nodupls = resolve_duplicates(
        eng_y9, ["upn", "year_end", "year_group", "school"], "eng_y9"
    )
    mat_y9 = pd.concat([mat_y9_17, mat_y9_19, mat_y9_21, mat_y9_22])
    mat_y9_nodupls = resolve_duplicates(
        mat_y9, ["upn", "year_end", "year_group", "school"], "mat_y9"
    )
    y9_non18 = eng_y9_nodupls.merge(
        mat_y9_nodupls,
        how="inner",
        on=["upn", "year_end", "year_group", "school"],
        validate="one_to_one",
    )
    y9_18 = resolve_duplicates(
        y7_to_y9_18[2],
        ["upn", "year_end", "year_group", "school"],
        ["mat_y9", "eng_y9"],
    )
    # Combine all Y9 data
    y9_raw = pd.concat([y9_non18, y9_18])
    # Resolve transfers
    y9_raw = resolve_duplicates(y9_raw, "upn", ["mat_y9", "eng_y9"])
    # Assign cohorts
    y9 = assign_cohort_info(y9_raw).drop(columns=["year_group", "repeat"])

    # Merge year 7 and year 8
    y7_y8 = y7.merge(
        y8, on=["upn"], how="outer", validate="one_to_one", suffixes=("_y7", "_y8")
    )
    y7_y8["school"] = y7_y8["school_y8"].combine_first(y7_y8["school_y7"])
    y7_y8["year_end"] = (
        y7_y8["year_end_y8"].combine_first(y7_y8["year_end_y7"]).astype(int)
    )
    y7_y8["repeat"] = False
    y7_y8.loc[y7_y8["repeat_y7"] | y7_y8["repeat_y8"], "repeat"] = True
    y7_y8["transfer"] = False
    y7_y8.loc[y7_y8["transfer_y7"] | y7_y8["transfer_y8"], "transfer"] = True
    # Check for inconsistent cohort assignments and drop them if there are less than 5
    is_inconsistent = ~(
        (y7_y8["cohort_y7"] == y7_y8["cohort_y8"])
        | y7_y8["cohort_y8"].isna()
        | y7_y8["cohort_y7"].isna()
    )
    if is_inconsistent.sum() > 5:
        raise ValueError(f"{is_inconsistent.sum()} UPNs have inconsistent cohorts.")
    y7_y8 = y7_y8[~is_inconsistent]
    y7_y8["cohort"] = y7_y8["cohort_y8"].combine_first(y7_y8["cohort_y7"]).astype(int)
    y7_y8 = y7_y8.drop(
        columns=[
            "repeat_y8",
            "repeat_y7",
            "school_y7",
            "year_end_y7",
            "school_y8",
            "year_end_y8",
            "cohort_y7",
            "cohort_y8",
            "transfer_y7",
            "transfer_y8",
        ]
    )
    # Add school information
    y7_y8 = y7_y8.merge(
        school_level,
        on=["school"],
        how="left",
        validate="many_to_one",
    )
    # Merge Y7-Y8 with Y9
    ks3 = y7_y8.merge(
        y9, on=["upn"], how="outer", validate="one_to_one", suffixes=("_y7y8", "_y9")
    )
    ks3["year_end"] = ks3["year_end_y7y8"].combine_first(ks3["year_end_y9"])
    ks3["cohort"] = ks3["cohort_y7y8"].combine_first(ks3["cohort_y9"])
    ks3["school"] = ks3["school_y7y8"].combine_first(ks3["school_y9"])
    ks3 = ks3.drop(
        columns=[
            "year_end_y7y8",
            "year_end_y9",
            "cohort_y7y8",
            "cohort_y9",
            "school_y7y8",
            "school_y9",
        ]
    )
    # Year 10
    # Year 10 2017
    eng_y10_17 = resolve_duplicates(eng_y10_17, ["upn", "year_end"], "eng_y10")
    mat_y10_17 = resolve_duplicates(mat_y10_17, ["upn", "year_end"], "mat_y10")
    y10_17 = eng_y10_17.merge(
        mat_y10_17,
        on=["upn", "year_end", "year_group"],
        how="outer",
        validate="one_to_one",
    )
    y10_17 = resolve_duplicates(y10_17, ["upn", "year_end"], ["eng_y10", "mat_y10"])
    # Year 10 2018
    y10_18["eng_y10"] = y10_18[["eng_lan_y10", "eng_lit_y10", "eng_lan_lit_y10"]].max(
        axis="columns"
    )
    y10_18 = resolve_duplicates(
        y10_18.drop(columns=(["eng_lan_y10", "eng_lit_y10", "eng_lan_lit_y10"])),
        ["upn", "year_end"],
        ["mat_y10", "eng_y10"],
    )
    # Y10 2019
    mat_y10_19["mat_y10"] = mat_y10_19[["mat_fou_y10", "mat_hig_y10"]].max(
        axis="columns"
    )
    mat_y10_19 = resolve_duplicates(
        mat_y10_19.drop(columns=(["mat_fou_y10", "mat_hig_y10"])),
        ["upn", "year_end"],
        "mat_y10",
    )
    eng_y10_19 = resolve_duplicates(eng_y10_19, ["upn", "year_end"], "eng_y10")
    y10_19 = mat_y10_19.merge(
        eng_y10_19,
        on=["upn", "year_end", "year_group"],
        how="outer",
        validate="one_to_one",
    )
    # Year 10 2021
    y10_21["mat_y10"] = y10_21[["mat_hig_y10", "mat_fou_y10"]].max(axis="columns")
    y10_21["eng_y10"] = y10_21[["eng_lan_y10", "eng_lit_y10"]].max(axis="columns")
    y10_21 = y10_21.drop(
        columns=["mat_hig_y10", "mat_fou_y10", "eng_lan_y10", "eng_lit_y10"]
    )
    y10_21 = resolve_duplicates(y10_21, ["upn", "year_end"], ["mat_y10", "eng_y10"])
    # Year 10 2022
    eng_lan_y10_22 = resolve_duplicates(
        eng_lan_y10_22, ["upn", "year_end"], "eng_lan_y10"
    )
    eng_lit_y10_22 = resolve_duplicates(
        eng_lit_y10_22, ["upn", "year_end"], "eng_lit_y10"
    )
    eng_y10_22 = eng_lan_y10_22.merge(
        eng_lit_y10_22,
        on=["upn", "year_end", "year_group"],
        how="outer",
        validate="one_to_one",
    )
    eng_y10_22["eng_y10"] = eng_y10_22[["eng_lit_y10", "eng_lan_y10"]].max(
        axis="columns"
    )
    eng_y10_22 = eng_y10_22.drop(columns=["eng_lit_y10", "eng_lan_y10"])
    mat_y10_22["mat_y10"] = mat_y10_22[["mat_hig_y10", "mat_fou_y10"]].max(
        axis="columns"
    )
    mat_y10_22 = mat_y10_22.drop(columns=["mat_hig_y10", "mat_fou_y10"])
    mat_y10_22 = resolve_duplicates(mat_y10_22, ["upn", "year_end"], "mat_y10")
    y10_22 = eng_y10_22.merge(
        mat_y10_22,
        on=["upn", "year_end", "year_group"],
        how="outer",
        validate="one_to_one",
    )
    y10_22 = resolve_duplicates(y10_22, ["upn", "year_end"], ["mat_y10", "eng_y10"])
    # Concatenate all Year 10
    y10_raw = pd.concat([y10_17, y10_18, y10_19, y10_21, y10_22])
    # Resolve transfers
    y10_raw = resolve_duplicates(y10_raw, "upn", ["mat_y10", "eng_y10"])
    # Assign cohorts
    y10 = assign_cohort_info(y10_raw).drop(columns=["year_group", "repeat"])

    # Merge KS3 with Y10
    ks3_y10 = ks3.merge(
        y10, on=["upn"], how="outer", validate="one_to_one", suffixes=("_ks3", "_y10")
    )
    ks3_y10["year_end"] = ks3_y10["year_end_ks3"].combine_first(ks3_y10["year_end_y10"])
    ks3_y10["cohort"] = ks3_y10["cohort_ks3"].combine_first(ks3_y10["cohort_y10"])
    ks3_y10 = ks3_y10.drop(
        columns=[
            "year_end_ks3",
            "year_end_y10",
            "cohort_ks3",
            "cohort_y10",
        ],
    )
    # GCSE
    school_map = school_names[["code", "school"]]

    gcse_18 = gcse_18.merge(school_map, how="left", on="code").drop(columns="code")
    gcse_21 = gcse_21.merge(school_map, how="left", on="code").drop(columns="code")
    gcse_raw = pd.concat([gcse_17, gcse_18, gcse_19, gcse_21, gcse_22])
    gcse_raw["school"] = gcse_raw["school"].apply(format_school)
    # Resolve transfers
    gcse = resolve_duplicates(gcse_raw, "upn", ["mat_y11", "eng_y11"])
    # Assign cohorts
    gcse = assign_cohort_info(gcse).drop(columns=["year_group", "repeat"])

    # Merge KS3-Y10 with GSCE
    ks3_ks4 = ks3_y10.merge(
        gcse, on=["upn"], how="outer", validate="one_to_one", suffixes=("", "_gcse")
    )
    ks3_ks4["year_end"] = ks3_ks4["year_end"].combine_first(ks3_ks4["year_end_gcse"])
    ks3_ks4["cohort"] = ks3_ks4["cohort"].combine_first(ks3_ks4["cohort_gcse"])
    ks3_ks4 = ks3_ks4.drop(columns=["year_end_gcse", "cohort_gcse"])

    # Pupil characteristics
    pupchar_raw = pd.concat(
        [pupchar_16, pupchar_17, pupchar_18, pupchar_21, pupchar_22, pupchar_23]
    )
    pupchar = resolve_duplicates(pupchar_raw, ["upn", "year_end"], ["is_white"])
    # Conduct
    sus_21_to_23 = pd.concat([sus_21, sus_22, sus_23])
    exp_21_to_23 = pd.concat([exp_21, exp_22, exp_23])
    sus_exp_21_to_23 = sus_21_to_23.merge(
        exp_21_to_23, on=["upn", "year_end"], how="outer", validate="one_to_one"
    )
    att_sus_exp_18 = att_18.merge(
        sus_exp_18, on=["upn", "year_end"], how="outer", validate="one_to_one"
    )
    att_sus_exp_16_to_18 = pd.concat([att_sus_exp_16, att_sus_exp_17, att_sus_exp_18])
    conduct = pd.concat([att_sus_exp_16_to_18, sus_exp_21_to_23])
    # Merge pupil characteristics and conduct
    pupchar_conduct = pupchar.merge(
        conduct, on=["upn", "year_end"], how="left", validate="one_to_one"
    )
    pupchar_conduct["imputed"] = False
    year_range = [
        x
        for x in range(
            int(ks3_ks4["year_end"].min()),
            int((ks3_ks4["year_end"].max() + 1)),
            1,
        )
    ]
    year_df = pd.DataFrame({"year_end": year_range})
    upn_df = pd.DataFrame({"upn": pupchar_conduct["upn"].unique().tolist()})
    dummy_df = year_df.merge(upn_df, how="cross")
    pupchar_conduct_imptd = dummy_df.merge(
        pupchar_conduct, on=["upn", "year_end"], how="left"
    ).reset_index(drop=True)
    pupchar_conduct_imptd["imputed"].fillna(True)
    ffill_cols = [c for c in pupchar_conduct_imptd.columns if c != "upn"]
    pupchar_conduct_imptd[ffill_cols] = pupchar_conduct_imptd.groupby("upn").ffill()
    pupchar_conduct_imptd[ffill_cols] = pupchar_conduct_imptd.groupby("upn").bfill()

    # Merge KS3-KS4 with pupil characteristics-conduct
    ks3_ks4_pupchar_conduct = ks3_ks4.merge(
        pupchar_conduct_imptd.drop(columns="school"),
        on=["upn", "year_end"],
        how="left",
        validate="one_to_one",
    )

    # Destinations
    dest_ks4_raw = pd.concat([dest_19_ks4, dest_21_ks4, dest_22[0]]).drop(
        columns=["dest_type", "school"]
    )
    dest_ks4 = (
        dest_ks4_raw.sort_values("year_end")
        .drop_duplicates(["upn"], keep="last")
        .drop(columns=["year_end"])
    )
    dest_ks5_raw = pd.concat([dest_19_ks5, dest_21_ks5, dest_22[1]]).drop(
        columns=["dest_type", "school"]
    )
    dest_ks5 = (
        dest_ks5_raw.sort_values("year_end")
        .drop_duplicates(["upn"], keep="last")
        .drop(columns=["year_end"])
    )
    dest = dest_ks4.merge(
        dest_ks5,
        on=["upn"],
        how="outer",
        suffixes=("_ks4", "_ks5"),
        validate="one_to_one",
    )

    is_int_ks4 = dest["dest_ks4"].apply(lambda x: isinstance(x, int)).sum()
    is_int_ks5 = dest["dest_ks5"].apply(lambda x: isinstance(x, int)).sum()
    if (is_int_ks4 + is_int_ks5) > 5:
        raise ValueError(
            f"{(is_int_ks4 + is_int_ks5)} UPNs have an integer as destination."
        )
    dest = dest[~dest["dest_ks4"].apply(lambda x: isinstance(x, int))]
    dest = dest[~dest["dest_ks5"].apply(lambda x: isinstance(x, int))]

    # Merge KS3-KS4-pupil characteristics-conduct with destinations
    full_df_raw = ks3_ks4_pupchar_conduct.merge(
        dest, on=["upn"], how="left", validate="one_to_one"
    )

    # Normalise
    full_df_pct = get_score_pct(full_df_raw, max_scores)

    # Select final columns
    full_df = full_df_pct[
        [
            "upn",
            "school",
            "cohort",
            "eng_y7",
            "mat_y7",
            "eng_y8",
            "mat_y8",
            "eng_y9",
            "mat_y9",
            "eng_y7_pct",
            "mat_y7_pct",
            "eng_y8_pct",
            "mat_y8_pct",
            "eng_y9_pct",
            "mat_y9_pct",
            "eng_y10",
            "mat_y10",
            "eng_y11",
            "mat_y11",
            "school_gcse",
            "dest_ks4",
            "dest_ks5",
            "repeat",
            "transfer",
            "suspensions",
            "expulsions",
            "attendance",
            "gender",
            "is_white",
            "eal",
            "sen",
            "in_care",
            "premium",
            "imputed",
            "local_auth",
            "north_south",
            "admissions_pol",
            "type",
            "phase",
            "ofstedrating",
            "cluster",
        ]
    ]

    # Save data
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    full_df_path_noext = os.path.join(PROCESSED_DIR, "full_df")
    logger.info(f"Saving data to {full_df_path_noext}.parquet")
    full_df.to_csv(f"{full_df_path_noext}.csv", index=False)
    full_df.to_parquet(f"{full_df_path_noext}.parquet")
