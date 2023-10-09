import pandas as pd

from uni_asp.preprocessing.clean_upn import clean_upn
from uni_asp.preprocessing.extract_col_name import extract_col_name
from uni_asp.preprocessing.format_ethnicity import format_ethnicity
from uni_asp.preprocessing.format_gender import format_gender
from uni_asp.preprocessing.format_incare import format_incare
from uni_asp.preprocessing.format_school import format_school
from uni_asp.preprocessing.format_sen import format_sen
from uni_asp.preprocessing.resolve_duplicates import resolve_duplicates


def format_columns(df, df_format, year_end=None, year_group=None, subject=None):
    """Takes a a raw dataframe and returns it with filtered columns and formated column names

    Args:
        df: a pandas dataframe or an excel file to be formated
        df_format: a string indicating the format of the dataframe. The following options are supported:
            "school_names" for school information from UL.
            "school_info" for school information from other sources.
            "pup_char" for pupil characteristics.
            "dest_type1" for destinations 2019 and 2021.
            "dest_type2" for destinations 2022.
            "att_exp_susp" for attendance and expulsions and suspensions 2017, 2018, 2020.
            "att" for attendance 2018, 2019, 2020.
            "susp" for suspensions 2021, 2022, 2023.
            "exp" for expulsions 2021, 2022, 2023.
            "ks3_type1" for years 7, 8 and 9 2017, 2019, 2021, 2022.
            "ks3_type2" for years 7, 8 and 9 2018. This will return three dataframes i.e. y7, y8 and y9. with indices 0, 1 and 2, respectively.
            "y10_2017" for year 10 2017.
            "y10_2018" for year 10 2018.
            "y10_2021" for year 10 2021.
            "y10_2022" for year 10 2022.
            "gcse_type1" for year 11 2017, 2018.
            "gcse_type2" for year 11 2019.
            "gcse_type3" for year 11 2021.
            "gcse_type4" for year 11.
        year_end: an integer indicating the end of the academic year for that particular dataframe
        year_group: an integer indicating the national curriculum year for that particular dataframe. This imput is required for grades/scores dataframes.
        subject: a string indicating the subject in the paritucalr dataframe. This imput is required for some grades/scores dataframes.

    Returns:
        A dataframe with filtered columns and formated column names
    """

    # set df_format == 'school_info' for school information dataframes
    if df_format == "school_names":
        # Identify relevant columns
        names = df.columns.tolist()
        school_col = extract_col_name(names, r"^school")
        code_col = extract_col_name(names, r"^internal")
        phase_col = extract_col_name(names, r"^phase")
        type_col = extract_col_name(names, r"^type")
        ns_col = extract_col_name(names, r"^north")
        cluster_col = extract_col_name(names, r"^cluster")

        # Create a dictionary to rename relevant columns
        new_cols = {
            school_col: "school",
            code_col: "code",
            phase_col: "phase",
            type_col: "type",
            ns_col: "north_south",
            cluster_col: "cluster",
        }

        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Format columns
        df["school"] = df["school"].apply(format_school)
        # Create a list to subselect relevant columns
        final_cols = [
            "school",
            "code",
            "phase",
            "type",
            "north_south",
            "cluster",
        ]

    # set df_format == 'school_info' for school information dataframes
    elif df_format == "school_info":
        # Identify relevant columns
        names = df.columns.tolist()
        school_col = extract_col_name(names, r"^schname")
        la_col = extract_col_name(names, r"^laname$")
        adm_col = extract_col_name(names, r"^admpol")
        ofr_col = extract_col_name(names, r"^ofstedrating")

        # Create a dictionary to rename relevant columns
        new_cols = {
            school_col: "school",
            la_col: "local_auth",
            adm_col: "admissions_pol",
            ofr_col: "ofstedrating",
        }
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # # Format columns
        df["school"] = df["school"].apply(format_school)
        # Create a list to subselect relevant columns
        final_cols = ["school", "local_auth", "admissions_pol", "ofstedrating"]

    # set df_format == 'pup_char' for pupil characteristicts dataframes
    elif df_format == "pup_char":
        # Identify relevant columns
        names = df.columns.tolist()
        school_col = extract_col_name(names, r"^school")
        upn_col = extract_col_name(names, r"^upn")
        gender_col = extract_col_name(names, r"sex")
        ethnicity_col = extract_col_name(names, r"ethni")
        premium_col = extract_col_name(names, r"^.*premium")
        sen_col = extract_col_name(names, r"^sen")
        eal_col = extract_col_name(names, r"^eal")
        in_care_col = extract_col_name(names, r"^looked")

        # Create a dictionary to rename relevant columns
        new_cols = {
            school_col: "school",
            upn_col: "upn",
            gender_col: "gender",
            ethnicity_col: "ethnicity",
            premium_col: "premium",
            sen_col: "sen",
            eal_col: "eal",
            in_care_col: "in_care",
        }
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Add year_end column
        df["year_end"] = year_end
        # Format columns
        df["school"] = df["school"].apply(format_school)
        df["gender"] = df["gender"].apply(format_gender)
        df["is_white"] = df["ethnicity"].apply(format_ethnicity)
        df = df.replace({"eal": {"Yes": True, "No": False}})
        df["sen"] = df["sen"].apply(format_sen)
        df["in_care"] = df["in_care"].apply(format_incare)
        df = df.replace({"premium": {"Yes": True, "No": False}})
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            "year_end",
            "school",
            "gender",
            "is_white",
            "eal",
            "sen",
            "in_care",
            "premium",
        ]

    # Set df_format == 'dest_type1' for 2019 and 2021 destinations dataframe
    elif df_format == "dest_type1":
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        if year_end == 2019:
            school_col = extract_col_name(names, r"^school$")
            dest_name_col = extract_col_name(names, r"detailed$")
        elif year_end == 2021:
            school_col = extract_col_name(names, r"^name")
            dest_name_col = extract_col_name(names, r"^destination")

        else:
            raise ValueError(f"Unsupported year_end with {df_format=}: {year_end!r}")
        # Create a dictionary to rename relevant columns
        new_cols = {school_col: "school", upn_col: "upn", dest_name_col: "dest"}
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Add destination_type column
        df["dest_type"] = year_group
        # Add year_end column
        df["year_end"] = year_end
        # Format school column
        df["school"] = df["school"].apply(format_school)
        # Create a list to subselect relevant columns
        final_cols = ["upn", "year_end", "dest", "dest_type", "school"]

    # Set df_format == 'dest_type2' for 2022 destinations dataframe
    elif df_format == "dest_type2":
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        year_group_col = extract_col_name(names, r"^nc_year")
        dest_type_col = extract_col_name(names, r"^destination_type")
        dest_name_col = extract_col_name(names, r"^destination_name")
        # Create a dictionary to rename relevant columns
        new_cols = {
            upn_col: "upn",
            year_group_col: "year_group",
            dest_type_col: "dest_type",
            dest_name_col: "dest",
        }
        # Create a list to subselect relevant columns
        final_cols = ["upn", "year_end", "dest", "dest_type"]
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Add year_end column
        df["year_end"] = year_end
        # Format year_group column
        df["year_group"] = df["year_group"].apply(
            lambda x: int(x.split("_")[-1]) if pd.notnull(x) else x
        )
        # subset for ks4 and drop missing UPNs
        df_ks4 = df[(df["year_group"] == 11) | (df["year_group"] == 12)].dropna(
            subset=["upn"]
        )
        # Filter for pupils with sixth form information
        sf_ks4 = df_ks4[df_ks4["dest_type"] == "Sixth form"]
        sf_ks4 = sf_ks4.drop_duplicates("upn")
        sf_ks4["priority"] = 3
        # Filter for pupils with pathway information
        pw_ks4 = df_ks4[df_ks4["dest_type"] == "Pathway"]
        pw_ks4 = pw_ks4.drop_duplicates("upn")
        pw_ks4["priority"] = 2
        # Filter for pupils with dash from information
        dash_ks4 = df_ks4[df_ks4["dest_type"] == "-"]
        dash_ks4 = dash_ks4.drop_duplicates("upn")
        dash_ks4["priority"] = 1
        # Create df for ks4 destinations
        df_ks4 = pd.concat([sf_ks4, pw_ks4, dash_ks4])
        df_ks4 = resolve_duplicates(df_ks4, "upn", "priority")
        df_ks4["dest_type"] = "ks4"
        df_ks4 = df_ks4[final_cols]
        df_ks4 = df_ks4.dropna(subset=["upn"]).drop_duplicates()
        df_ks4 = clean_upn(df_ks4)

        # subset for ks5 and drop missing UPNs
        df_ks5 = df[(df["year_group"] == 13) | (df["year_group"] == 14)].dropna(
            subset=["upn"]
        )
        # Filter for pupils with sixth form information
        sf_ks5 = df_ks5[df_ks5["dest_type"] == "Sixth form"]
        sf_ks5 = sf_ks5.drop_duplicates("upn")
        sf_ks5["priority"] = 3
        # Filter for pupils with pathway information
        pw_ks5 = df_ks5[df_ks5["dest_type"] == "Pathway"]
        pw_ks5 = pw_ks5.drop_duplicates("upn")
        pw_ks5["priority"] = 2
        # Filter for pupils with dash from information
        dash_ks5 = df_ks5[df_ks5["dest_type"] == "-"]
        dash_ks5 = dash_ks5.drop_duplicates("upn")
        dash_ks5["priority"] = 1
        # Create df for ks4 destinations
        df_ks5 = pd.concat([sf_ks5, pw_ks5, dash_ks4])
        df_ks5 = resolve_duplicates(df_ks5, "upn", "priority")
        df_ks5["dest_type"] = "ks5"
        df_ks5 = df_ks5[final_cols]
        df_ks5 = df_ks5.dropna(subset=["upn"]).drop_duplicates()
        df_ks5 = clean_upn(df_ks5)

    # Set df_format == 'att_expuls' for attendance+expulsions 2016-2018 dataframes
    elif df_format == "att_exp_susp":
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        suspen_col = extract_col_name(names, r"period exclusions")
        expuls_col = extract_col_name(names, r"permanent exclusions")
        if year_end != 2018:
            att_col = extract_col_name(names, r"\battendance\b")
            # Create a dictionary to rename relevant columns
            new_cols = {
                upn_col: "upn",
                suspen_col: "suspensions",
                expuls_col: "expulsions",
                att_col: "attendance",
            }
            # Rename relevant columns
            df = df.rename(columns=new_cols)
            # Format attendance column
            df["attendance"] = df["attendance"].apply(
                lambda x: float(x.replace("%", "")) * 0.01 if pd.notnull(x) else x
            )
            # Add suspensions and expulsions accross upns
            sus_exp = (
                df.groupby(by=["upn"])[["suspensions", "expulsions"]]
                .sum()
                .reset_index()
            )
            # Take average attendance acrross upns
            att = df.groupby(by=["upn"])["attendance"].mean().reset_index()
            df = df.drop_duplicates(["upn"])
            df = sus_exp.merge(att, on=["upn"], validate="one_to_one")
            # Create a list to subselect relevant columns
            final_cols = ["upn", "year_end", "suspensions", "expulsions", "attendance"]
        else:
            # Create a dictionary to rename relevant columns
            new_cols = {
                upn_col: "upn",
                suspen_col: "suspensions",
                expuls_col: "expulsions",
            }
            # Rename relevant columns
            df = df.rename(columns=new_cols)
            # Add suspensions and expulsions accross upns
            df = (
                df.groupby(by=["upn"])[["suspensions", "expulsions"]]
                .sum()
                .reset_index()
            )
            # Create a list to subselect relevant columns
            final_cols = ["upn", "year_end", "suspensions", "expulsions"]
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Conver columns to integers
        df["suspensions"] = df["suspensions"].astype(int)
        df["expulsions"] = df["expulsions"].astype(int)
        # Add year_end column
        df["year_end"] = year_end

    # Set df_format == 'att' for attendance 2018-2020 dataframes
    elif df_format == "att":
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^student")
        pres_col = extract_col_name(names, r"^present")
        auth_col = extract_col_name(names, r"^auth")
        unauth_col = extract_col_name(names, r"^unauth")
        poss_col = extract_col_name(names, r"^possible")
        # Create a dictionary to rename relevant columns
        new_cols = {
            upn_col: "upn",
            pres_col: "present",
            auth_col: "authorised",
            unauth_col: "unauthorised",
            poss_col: "total",
        }
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Add year_end column
        df["year_end"] = year_end
        # Add values accross upn's with more than one school
        att = (
            df.groupby(by=["upn"])[["present", "authorised", "unauthorised", "total"]]
            .sum()
            .reset_index()
        )
        # Compute percent attendance
        att["attendance"] = (att["present"] + att["authorised"]) / att["total"]
        df = df.drop_duplicates(["upn", "year_end"])
        df = df[["upn", "year_end"]].merge(att, on=["upn"], validate="one_to_one")
        # Create a list to subselect relevant columns
        final_cols = ["upn", "year_end", "attendance"]

    # Set df_format == 'expuls' for expulsions 2021-2023 dataframes
    elif df_format == "exp":
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        expuls_col = extract_col_name(names, r"permanent exclusions")
        # Create a dictionary to rename relevant columns
        new_cols = {upn_col: "upn", expuls_col: "expulsions"}
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Add suspensions and expulsions accross upns
        df = df.groupby(by=["upn"])["expulsions"].sum().reset_index()
        # Add year_end column
        df["year_end"] = year_end
        # Conver columns to integers
        df["expulsions"] = df["expulsions"].astype(int)
        # Create a list to subselect relevant columns
        final_cols = ["upn", "year_end", "expulsions"]

    # Set df_format == 'suspens' for supensions 2021-2023 dataframes
    elif df_format == "susp":
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        suspen_col = extract_col_name(names, r"period exclusions")
        # Create a dictionary to rename relevant columns
        new_cols = {upn_col: "upn", suspen_col: "suspensions"}
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Add suspensions and expulsions accross upns
        df = df.groupby(by=["upn"])["suspensions"].sum().reset_index()
        # Add year_end column
        df["year_end"] = year_end
        # Conver columns to integers
        df["suspensions"] = df["suspensions"].astype(int)
        # Create a list to subselect relevant columns
        final_cols = ["upn", "year_end", "suspensions"]

    # Set df_format == 'ks3_type1' for KS3 2017, 2019, 2021 and 2022 dataframes
    elif df_format == "ks3_type1":
        # Raise errors if required imputs are not provided
        if subject is None:
            raise ValueError("Please, provide subject")
        if subject != "mat" and subject != "eng":
            raise ValueError(
                f"subject not supported. Please set subject equal to 'mat' or 'eng'. "
                f"Got {subject!r}"
            )

        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        school_col = extract_col_name(names, r"^school")
        if year_end == 2017:
            if subject == "eng" and year_group == 9:
                grade_col = extract_col_name(names, r"^total.1")
            else:
                grade_col = extract_col_name(names, r"^total")
        elif year_end == 2019:
            if subject == "eng" and (year_group == 7 or year_group == 8):
                grade_col = extract_col_name(names, r"^total\nexc")
            else:
                grade_col = extract_col_name(names, r"^overall")
        elif year_end == 2021:
            if subject == "eng" and year_group == 7:
                grade_col = extract_col_name(names, r"^total")
            else:
                grade_col = extract_col_name(names, r"^overall total")
        elif year_end == 2022:
            grade_col = extract_col_name(names, r"^total")
        else:
            raise ValueError(f"Unsupported year_end with {df_format=}: {year_end!r}")
        # Create a dictionary to rename relevant columns
        col = subject + "_y" + str(year_group)
        # Create a dictionary to rename relvant columns
        new_cols = {
            upn_col: "upn",
            school_col: "school",
            grade_col: col,
        }
        # Rename relevant columns names
        df = df.rename(columns=new_cols)
        # Format school column
        df["school"] = df["school"].apply(format_school)
        # Convert grade column to float
        df = df.dropna(subset=["upn"])
        df[col] = df[col].astype("float")
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            "year_end",
            "year_group",
            "school",
            col,
        ]

    # Set df_format == 'ks3_type2' for KS3 2018 data
    elif df_format == "ks3_type2":
        # Select relevant columns for year 7 math and english and merge dataframes
        for key in ["Y7_Eng", "Y7_Mat"]:
            curr_subject = key[3:6].lower() + "_" + key[:2].lower()
            curr_df = df[key].rename(
                {"UPN": "upn", "TOTAL": curr_subject, "School": "school"}, axis=1
            )
            # Format school column
            curr_df["school"] = curr_df["school"].apply(format_school)
            # Resolver duplicates
            curr_df = resolve_duplicates(curr_df, ["upn", "school"], curr_subject)
            if key == "Y7_Eng":
                y7_df = curr_df[["upn", "school", curr_subject]]
            else:
                y7_df = y7_df.merge(
                    curr_df[["upn", "school", curr_subject]],
                    how="outer",
                    on=["upn", "school"],
                    validate="one_to_one",
                )
        # Add year_end column
        y7_df["year_end"] = year_end
        # Add year_group column
        y7_df["year_group"] = 7
        # Format school column
        y7_df["school"] = y7_df["school"].apply(format_school)
        # Convert grade columns to float
        y7_df = y7_df.dropna(subset=["upn"])
        y7_df["eng_y7"] = y7_df["eng_y7"].astype("float")
        y7_df["mat_y7"] = y7_df["mat_y7"].astype("float")
        # Subselect relevant columns
        y7_df = y7_df[["upn", "year_end", "year_group", "school", "eng_y7", "mat_y7"]]
        y7_df = y7_df.dropna(subset=["upn"]).drop_duplicates()
        y7 = clean_upn(y7_df)

        # Select relevant columns for year 8 math and english and merge dataframes
        for key in ["Y8_Eng", "Y8_Mat"]:
            curr_subject = key[3:6].lower() + "_" + key[:2].lower()

            curr_df = df[key].rename(
                {"UPN": "upn", "TOTAL": curr_subject, "School": "school"}, axis=1
            )
            # Format school column
            curr_df["school"] = curr_df["school"].apply(format_school)
            # Resolver duplicates
            curr_df = resolve_duplicates(curr_df, ["upn", "school"], curr_subject)
            if key == "Y8_Eng":
                y8_df = curr_df[["upn", "school", curr_subject]]
            else:
                y8_df = y8_df.merge(
                    curr_df[["upn", "school", curr_subject]],
                    how="outer",
                    on=["upn", "school"],
                    validate="one_to_one",
                )
        # Add year_end column
        y8_df["year_end"] = year_end
        # Add year_group column
        y8_df["year_group"] = 8
        # Format school column
        y8_df["school"] = y8_df["school"].apply(format_school)
        # Convert grade columns to float
        y8_df = y8_df.dropna(subset=["upn"])
        y8_df["eng_y8"] = y8_df["eng_y8"].astype("float")
        y8_df["mat_y8"] = y8_df["mat_y8"].astype("float")
        # Subselect relevant columns
        y8_df = y8_df[["upn", "year_end", "year_group", "school", "eng_y8", "mat_y8"]]
        y8_df = y8_df.dropna(subset=["upn"]).drop_duplicates()
        y8 = clean_upn(y8_df)

        # Select relevant columns for year 9 math and english and merge dataframes
        for key in ["Y9_Eng", "Y9_Mat"]:
            curr_subject = key[3:6].lower() + "_" + key[:2].lower()
            curr_df = df[key].rename(
                {"UPN": "upn", "TOTAL": curr_subject, "School": "school"}, axis=1
            )
            # Format school column
            curr_df["school"] = curr_df["school"].apply(format_school)
            # Resolver duplicates
            curr_df = resolve_duplicates(curr_df, ["upn", "school"], curr_subject)
            if key == "Y9_Eng":
                y9_df = curr_df[["upn", "school", curr_subject]]
            else:
                y9_df = y9_df.merge(
                    curr_df[["upn", "school", curr_subject]],
                    how="outer",
                    on=["upn", "school"],
                    validate="one_to_one",
                )
        # Add year_end column
        y9_df["year_end"] = year_end
        # Add year_group column
        y9_df["year_group"] = 9
        # Format school column
        y9_df["school"] = y9_df["school"].apply(format_school)
        # Convert grade columns to float
        y9_df = y9_df.dropna(subset=["upn"])
        y9_df["eng_y9"] = y9_df["eng_y9"].astype("float")
        y9_df["mat_y9"] = y9_df["mat_y9"].astype("float")
        # Subselect relevant columns
        y9_df = y9_df[["upn", "year_end", "year_group", "school", "eng_y9", "mat_y9"]]
        y9_df = y9_df.dropna(subset=["upn"]).drop_duplicates()
        y9 = clean_upn(y9_df)

    # Set df_format == 'y10_2017' for 2017 year 10 dataframe
    elif df_format == "y10_2017":
        # Raise errors if required imputs are not provided
        if subject is None:
            raise ValueError("Please, provide subject")
        if subject != "mat" and subject != "eng":
            raise ValueError(
                f"subject not supported. Please set subject equal to 'mat' or 'eng'. "
                f"Got {subject!r}"
            )
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        eng_col = extract_col_name(names, r"^projected gcse")
        # Create column name for subject
        col = subject + "_y" + str(year_group)
        # Create a dictionaty to rename relevant columns
        new_cols = {upn_col: "upn", eng_col: col}
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Reformat
        df = df.replace({col: {"U": 0, "u": 0}})
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Convert grade columns to float
        df = df.dropna(subset=["upn"])
        df[col] = df[col].astype("float")
        # Create a list to subselect relevant columns
        final_cols = ["upn", "year_end", "year_group", subject + "_y" + str(year_group)]

    # Set df_format == 'y10_2018' for 2018 year 10 dataframe
    elif df_format == "y10_2018":
        # Raise errors if required imputs are not provided
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Identigy relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"upn")
        eng_lit_col = extract_col_name(names, r"english literature")
        eng_lang_col = extract_col_name(names, r"english language$")
        eng_lang_lit_col = extract_col_name(names, r"^english language & lit")
        mat_col = extract_col_name(names, r"^mathematics")
        # Create a dictionary to rename relevant columns
        new_cols = {
            upn_col: "upn",
            eng_lit_col: "eng_lit_y10",
            eng_lang_col: "eng_lan_y10",
            eng_lang_lit_col: "eng_lan_lit_y10",
            mat_col: "mat_y10",
        }
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        df = df.replace(
            {
                "eng_lit_y10": {"U": 0, "u": 0},
                "eng_lan_y10": {"U": 0, "u": 0},
                "eng_lan_lit_y10": {"U": 0, "u": 0},
                "mat_y10": {"U": 0, "u": 0},
            }
        )
        # Convert grade columns to float
        df = df.dropna(subset=["upn"])
        df["eng_lit_y10"] = df["eng_lit_y10"].astype("float")
        df["eng_lan_y10"] = df["eng_lan_y10"].astype("float")
        df["eng_lan_lit_y10"] = df["eng_lan_lit_y10"].astype("float")
        df["mat_y10"] = df["mat_y10"].astype("float")
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            "year_end",
            "year_group",
            "eng_lan_y10",
            "eng_lit_y10",
            "eng_lan_lit_y10",
            "mat_y10",
        ]

    # Set df_format == 'y10_2019' for 2019 year 10 dataframe
    elif df_format == "y10_2019":
        # Raise errors if required imputs are not provided
        if subject is None:
            raise ValueError("Please, provide subject")
        if subject != "mat" and subject != "eng":
            raise ValueError(
                f"subject not supported. Please set subject equal to 'mat' or 'eng'. "
                f"Got {subject!r}"
            )

        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Format english
        if subject == "eng":
            # Identify relevant columns
            names = df.columns.tolist()
            upn_col = extract_col_name(names, r"^upn")
            eng_col = extract_col_name(names, r"^working at")
            # Create a dicitonary to rename relevant columns
            new_cols = {upn_col: "upn", eng_col: "eng_y10"}
            # Rename relvant columns
            df = df.rename(columns=new_cols)
            df = df.replace({"eng_y10": {"U": 0, "u": 0}})
            # Convert grade columns to float
            df = df.dropna(subset=["upn"])
            df["eng_y10"] = df["eng_y10"].astype("float")
            # Add year_end column
            df["year_end"] = year_end
            # Add year_group column
            df["year_group"] = year_group
            # Create a list to subselect relevant columns
            final_cols = ["upn", "year_end", "year_group", "eng_y10"]
        # Format math
        elif subject == "mat":
            for key in ["Data FND", "Data HGH"]:
                curr_df = df[key]
                if key == "Data FND":
                    curr_df = curr_df.rename(
                        {"UPN": "upn", "AGE RELATED GRADE": "mat_fou_y10"}, axis=1
                    )
                    curr_df = curr_df.replace({"mat_fou_y10": {"U": 0, "u": 0}})
                    curr_df = resolve_duplicates(curr_df, "upn", "mat_fou_y10").dropna(
                        subset=["upn"]
                    )
                    curr_df["mat_fou_y10"] = curr_df["mat_fou_y10"].astype("float")
                    merged_df = curr_df[["upn", "mat_fou_y10"]]
                elif key == "Data HGH":
                    curr_df = curr_df.rename(
                        {"UPN": "upn", "AGE RELATED GRADE": "mat_hig_y10"}, axis=1
                    )
                    curr_df = curr_df.replace({"mat_hig_y10": {"U": 0, "u": 0}})
                    curr_df = resolve_duplicates(curr_df, "upn", "mat_hig_y10").dropna(
                        subset=["upn"]
                    )
                    curr_df["mat_hig_y10"] = curr_df["mat_hig_y10"].astype("float")
                    merged_df = merged_df.merge(
                        curr_df[["upn", "mat_hig_y10"]],
                        how="outer",
                        on="upn",
                        validate="one_to_one",
                    )
            df = merged_df
            # Add year_end column
            df["year_end"] = year_end
            # Add year_group column
            df["year_group"] = year_group
            # Create a list to subselect relevant columns
            final_cols = ["upn", "year_end", "year_group", "mat_fou_y10", "mat_hig_y10"]

    # Set df_format == 'y10_2021' for 2021 year 10 dataframe
    elif df_format == "y10_2021":
        # Raise errors if required imputs are not provided
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Go through the different sheets and combine math and english grades into one dataframe
        for key in ["EngLit", "EngLang", "Maths_higher", "Maths_foundation"]:
            curr_df = df[key]
            if key == "EngLit":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "GCSE grade\n(CALC bd)": "eng_lit_y10"}, axis=1
                )
                curr_df = curr_df.replace({"eng_lit_y10": {"U": 0, "u": 0}})
                curr_df = resolve_duplicates(curr_df, "upn", "eng_lit_y10").dropna(
                    subset=["upn"]
                )
                curr_df["eng_lit_y10"] = curr_df["eng_lit_y10"].astype("float")
                merged_df = curr_df[["upn", "eng_lit_y10"]]
            elif key == "EngLang":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "GCSE grade\n(CALC bd)": "eng_lan_y10"}, axis=1
                )
                curr_df = curr_df.replace({"eng_lan_y10": {"U": 0, "u": 0}})
                curr_df = resolve_duplicates(curr_df, "upn", "eng_lan_y10").dropna(
                    subset=["upn"]
                )
                curr_df["eng_lan_y10"] = curr_df["eng_lan_y10"].astype("float")
                merged_df = merged_df.merge(
                    curr_df[["upn", "eng_lan_y10"]],
                    how="outer",
                    on="upn",
                    validate="one_to_one",
                )
            elif key == "Maths_higher":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "GCSE grade\n(CALC bd)": "mat_hig_y10"}, axis=1
                )
                curr_df = curr_df.replace({"mat_hig_y10": {"U": 0, "u": 0}})
                curr_df = resolve_duplicates(curr_df, "upn", "mat_hig_y10").dropna(
                    subset=["upn"]
                )
                curr_df["mat_hig_y10"] = curr_df["mat_hig_y10"].astype("float")
                merged_df = merged_df.merge(
                    curr_df[["upn", "mat_hig_y10"]],
                    how="outer",
                    on="upn",
                    validate="one_to_one",
                )
            elif key == "Maths_foundation":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "GCSE grade\n(CALC bd)": "mat_fou_y10"}, axis=1
                )
                curr_df = curr_df.replace({"mat_fou_y10": {"U": 0, "u": 0}})
                curr_df = resolve_duplicates(curr_df, "upn", "mat_fou_y10").dropna(
                    subset=["upn"]
                )
                curr_df["mat_fou_y10"] = curr_df["mat_fou_y10"].astype("float")
                merged_df = merged_df.merge(
                    curr_df[["upn", "mat_fou_y10"]],
                    how="outer",
                    on="upn",
                    validate="one_to_one",
                )
        df = merged_df
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            "year_end",
            "year_group",
            "eng_lit_y10",
            "eng_lan_y10",
            "mat_hig_y10",
            "mat_fou_y10",
        ]

    # Set df_format == 'y10_2022' for 2022 year 10 dataframe
    elif df_format == "y10_2022":
        # Raise errors if required imputs are not provided
        if subject is None:
            raise ValueError("Please, provide subject")
        if subject != "mat" and subject != "eng_lan" and subject != "eng_lit":
            raise ValueError(
                f"subject not supported. Please set subject equal to 'mat' or 'eng'. "
                f"Got {subject!r}"
            )
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Format english
        if subject[:3].lower() == "eng":
            # Identify relevant columns
            names = df.columns.tolist()
            upn_col = extract_col_name(names, r"^upn")
            eng_col = extract_col_name(names, r"modelled grade")
            # Create column name for subject
            col = subject + "_y" + str(year_group)
            # Create a dictionary to rename relevant columns
            new_cols = {upn_col: "upn", eng_col: col}
            # Rename relevant column
            df = df.rename(columns=new_cols)
            df = df.replace({col: {"U": 0, "u": 0}})
            # Convert grade columns to ineger
            df = df.dropna(subset=["upn"])
            df[col] = df[col].astype("float")
            # Add year_end column
            df["year_end"] = year_end
            # Add year_group column
            df["year_group"] = year_group
            # Create a list to subselect relevant columns
            final_cols = [
                "upn",
                "year_end",
                "year_group",
                subject + "_y" + str(year_group),
            ]
        # Format math
        elif subject == "mat":
            for key in ["FOUNDATION", "HIGHER"]:
                curr_df = df[key]
                if key == "FOUNDATION":
                    curr_df = curr_df.rename(
                        {"UPN": "upn", "Modelled grade": "mat_fou_y10"}, axis=1
                    )
                    curr_df = curr_df.replace({"mat_fou_y10": {"U": 0, "u": 0}})
                    curr_df = resolve_duplicates(curr_df, "upn", "mat_fou_y10").dropna(
                        subset=["upn"]
                    )
                    curr_df["mat_fou_y10"] = curr_df["mat_fou_y10"].astype("float")
                    merged_df = curr_df[["upn", "mat_fou_y10"]]
                elif key == "HIGHER":
                    curr_df = curr_df.rename(
                        {"UPN": "upn", "Modelled grade": "mat_hig_y10"}, axis=1
                    )
                    curr_df = curr_df.replace({"mat_hig_y10": {"U": 0, "u": 0}})
                    curr_df = resolve_duplicates(curr_df, "upn", "mat_hig_y10").dropna(
                        subset=["upn"]
                    )
                    curr_df["mat_hig_y10"] = curr_df["mat_hig_y10"].astype("float")
                    merged_df = merged_df.merge(
                        curr_df[["upn", "mat_hig_y10"]],
                        how="outer",
                        on="upn",
                        validate="one_to_one",
                    )
            df = merged_df
            # Add year_end column
            df["year_end"] = year_end
            # Add year_group column
            df["year_group"] = year_group
            # Create a list to subselect relevant columns
            final_cols = ["upn", "year_end", "year_group", "mat_fou_y10", "mat_hig_y10"]

    # Set  df_format == 'gcse_type1' for 2017 and 2018 gcse dataframes
    elif df_format == "gcse_type1":
        # Raise errors if required imputs are not provided
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        eng_lang_col = extract_col_name(names, r"^english lang")
        eng_lit_col = extract_col_name(names, r"^english lit")
        mat_col = extract_col_name(names, r"^maths")
        if year_end == 2017:
            school_col = extract_col_name(names, r"^school")
            string_school = "school"
        elif year_end == 2018:
            school_col = extract_col_name(names, r"^short code")
            string_school = "code"
        # Create a dictionary to rename relevant columns
        new_cols = {
            upn_col: "upn",
            eng_lang_col: "eng_lan_y11",
            eng_lit_col: "eng_lit_y11",
            mat_col: "mat_y11",
            school_col: string_school,
        }
        # Rename relevant column
        df = df.rename(columns=new_cols)
        # Replace U and X with 0 and nan
        df = df.replace(
            {
                "eng_lan_y11": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")},
                "eng_lit_y11": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")},
                "mat_y11": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")},
            }
        )
        # Convert grade columns to floats
        df = df.dropna(subset=["upn"])
        df["eng_lan_y11"] = df["eng_lan_y11"].astype("float")
        df["eng_lit_y11"] = df["eng_lit_y11"].astype("float")
        df["mat_y11"] = df["mat_y11"].astype("float")
        # Create english column as the highest score between lang and lit
        df["eng_y11"] = df[["eng_lan_y11", "eng_lit_y11"]].max(axis="columns")
        # Add year_end colum
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            string_school,
            "year_end",
            "year_group",
            "eng_y11",
            "mat_y11",
        ]

    # Set  df_format == 'gcse_type2' for 2019 gcse scores
    elif df_format == "gcse_type2":
        # Raise errors if required imputs are not provided
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Identify relevant columns
        names = df.columns.tolist()
        upn_col = extract_col_name(names, r"^upn")
        eng_col = extract_col_name(names, r"^a8 score english")
        mat_col = extract_col_name(names, r"^a8 score math")
        school_col = extract_col_name(names, r"^school")
        # Create a dicitonary to rename relevant columns
        new_cols = {
            upn_col: "upn",
            eng_col: "eng_y11",
            mat_col: "mat_y11",
            school_col: "school",
        }
        # Rename relevant columns
        df = df.rename(columns=new_cols)
        # Format grade columns
        df = df.replace(
            {
                "eng_y11": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")},
                "mat_y11": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")},
            }
        )
        # Convert grade columns to floats
        df = df.dropna(subset=["upn"])
        df["eng_y11"] = df["eng_y11"].astype("float").div(2)
        df["mat_y11"] = df["mat_y11"].astype("float").div(2)
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = ["upn", "school", "year_end", "year_group", "eng_y11", "mat_y11"]

    # Set  df_format == 'gcse_type3' for 2021 gcse scores
    elif df_format == "gcse_type3":
        # Raise errors if required imputs are not provided
        if year_group is None:
            raise ValueError("Please, provide year_group")
        # Replace U grade with 0
        df = df.replace(
            {"Grade": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")}}
        )
        # Go through the different sheets and combine math and english scores into one dataframe
        for curr_subject in [
            "GCSE English Language",
            "GCSE English Literature",
            "GCSE Mathematics",
        ]:
            curr_df = df[["UPN", "Grade", "Short code"]][df["Subject"] == curr_subject]
            if curr_subject == "GCSE English Language":
                merged_df = curr_df.rename(
                    {"UPN": "upn", "Grade": "eng_lan_y11", "Short code": "code"}, axis=1
                )
                merged_df = resolve_duplicates(
                    merged_df, ["upn", "code"], "eng_lan_y11"
                ).dropna(subset=["upn"])
                merged_df["eng_lan_y11"] = merged_df["eng_lan_y11"].astype("float")
            elif curr_subject == "GCSE English Literature":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "Grade": "eng_lit_y11", "Short code": "code"}, axis=1
                ).dropna(subset=["upn"])
                curr_df = resolve_duplicates(curr_df, ["upn", "code"], "eng_lit_y11")
                curr_df["eng_lit_y11"] = curr_df["eng_lit_y11"].astype("float")
                merged_df = merged_df.merge(
                    curr_df, how="outer", on=["upn", "code"], validate="one_to_one"
                )
            elif curr_subject == "GCSE Mathematics":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "Grade": "mat_y11", "Short code": "code"}, axis=1
                ).dropna(subset=["upn"])
                curr_df = resolve_duplicates(curr_df, ["upn", "code"], "mat_y11")
                curr_df["mat_y11"] = curr_df["mat_y11"].astype("float")
                merged_df = merged_df.merge(
                    curr_df, how="outer", on=["upn", "code"], validate="one_to_one"
                )
        df = merged_df
        # Create english column as the highest score between lang and lit
        df["eng_y11"] = df[["eng_lan_y11", "eng_lit_y11"]].max(axis="columns")
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            "code",
            "year_end",
            "year_group",
            "eng_y11",
            "mat_y11",
        ]

    # Set  df_format == 'gcse_type4' for 2022 gcse scores
    elif df_format == "gcse_type4":
        # Format subject column
        df = df.replace(
            {
                "Subject": {
                    "English Literature": "eng_lit",
                    "English Literature grade": "eng_lit",
                    "Mathematics": "math",
                    "Maths grade": "math",
                    "English Language": "eng_lan",
                    "English Language grade": "eng_lan",
                },
                "Grade": {"U": 0, "u": 0, "X": float("nan"), "x": float("nan")},
            }
        )
        # Go through the different sheets and combine english and math scores into one dataframe
        for curr_subject in ["eng_lang", "eng_lit", "math"]:
            curr_df = df[["UPN", "Grade", "School Name"]][df["Subject"] == curr_subject]
            if curr_subject == "eng_lang":
                merged_df = curr_df.rename(
                    {"UPN": "upn", "Grade": "eng_lan_y11", "School Name": "school"},
                    axis=1,
                ).dropna(subset=["upn"])
                # Convert grade columns to floats
                merged_df["eng_lan_y11"] = merged_df["eng_lan_y11"].astype("float")
                # Resolve duplicates
                merged_df = resolve_duplicates(
                    merged_df, ["upn", "school"], "eng_lan_y11"
                )
            elif curr_subject == "eng_lit":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "Grade": "eng_lit_y11", "School Name": "school"},
                    axis=1,
                ).dropna(subset=["upn"])
                # Convert grade columns to floats
                curr_df["eng_lit_y11"] = curr_df["eng_lit_y11"].astype("float")
                # Resolve duplicates
                curr_df = resolve_duplicates(curr_df, ["upn", "school"], "eng_lit_y11")
                merged_df = merged_df.merge(
                    curr_df, how="outer", on=["upn", "school"], validate="one_to_one"
                )
            elif curr_subject == "math":
                curr_df = curr_df.rename(
                    {"UPN": "upn", "Grade": "mat_y11", "School Name": "school"}, axis=1
                ).dropna(subset=["upn"])
                # Convert grade columns to floats
                curr_df["mat_y11"] = curr_df["mat_y11"].astype("float")
                # Resolve duplicates
                curr_df = resolve_duplicates(curr_df, ["upn", "school"], "mat_y11")
                merged_df = merged_df.merge(
                    curr_df, how="outer", on=["upn", "school"], validate="one_to_one"
                )
        df = merged_df
        # Create english column as the highest score between lang and lit
        df["eng_y11"] = df[["eng_lan_y11", "eng_lit_y11"]].max(axis="columns")
        # Add year_end column
        df["year_end"] = year_end
        # Add year_group column
        df["year_group"] = year_group
        # Create a list to subselect relevant columns
        final_cols = [
            "upn",
            "school",
            "year_end",
            "year_group",
            "eng_y11",
            "mat_y11",
        ]

    # Raise error if defined df_format is not supported
    else:
        raise ValueError(f"df_format not supported" f"Got {subject!r}")

    # Prepare final df
    if df_format == "ks3_type2":
        new_df = [y7, y8, y9]
    elif df_format == "dest_type2":
        new_df = [df_ks4, df_ks5]
    elif (df_format == "school_info") | (df_format == "school_names"):
        new_df = df[final_cols].drop_duplicates()
    else:
        new_df = df[final_cols]
        new_df = new_df.dropna(subset=["upn"]).drop_duplicates()
        new_df = clean_upn(new_df)
    return new_df
