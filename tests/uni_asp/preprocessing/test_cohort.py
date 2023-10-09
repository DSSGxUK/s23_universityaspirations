import io
import textwrap

import pandas as pd

from tests.utils import assert_frame_equal
from uni_asp.preprocessing.cohort import (
    assign_cohort_info,
    assign_cohorts,
    resolve_repeated_yeargroups,
)


def test_repetition_drop():
    df = to_df(
        """
            upn, year_group, year_end
            aaa,          7,     2018
            aaa,          7,     2019
            bbb,          7,     2017
            bbb,          8,     2018
            ccc,          7,     2018
            ccc,          8,     2018
            """
    )

    out_df = resolve_repeated_yeargroups(df)

    expected_df = to_df(
        """
        upn, year_group, year_end, repeat
        aaa,          7,     2019, True
        bbb,          7,     2017, False
        bbb,          8,     2018, False
        ccc,          7,     2018, False
        ccc,          8,     2018, False
        """
    )

    assert_frame_equal(out_df, expected_df)


def test_assign_cohorts():
    df = to_df(
        """
        upn, year_group, year_end
        aaa,          7,     2018
        bbb,          8,     2018
        ccc,          7,     2017
        """
    )

    out_df = assign_cohorts(df)

    assert out_df["cohort"].tolist() == [5, 4, 4]


def test_assign_cohorts_without_upns():
    df = to_df(
        """
        year_group, year_end
                 7,     2018
                 8,     2018
                 7,     2017
        """
    )

    out_df = assign_cohorts(df, upn=False)

    assert out_df["cohort"].tolist() == [5, 4, 4]


def test_assign_cohort_info():
    df = to_df(
        """
        upn, year_group, year_end
        aaa,          7,     2018
        aaa,          7,     2019
        bbb,          7,     2017
        bbb,          8,     2018
        ccc,          9,     2017
        ccc,          10,    2018
        ddd,          10,    2022
        """
    )

    out_df = assign_cohort_info(df)

    expected_df = to_df(
        """
        upn, year_group, year_end,  repeat,  cohort
        aaa,          7,     2019,    True,       6
        bbb,          7,     2017,   False,       4
        bbb,          8,     2018,   False,       4
        ccc,          9,     2017,   False,       2
        ccc,          10,    2018,   False,       2
        ddd,          10,    2022,   False,       6
        """
    )

    assert_frame_equal(out_df, expected_df)


def to_df(csv: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(textwrap.dedent(csv)),
        skipinitialspace=True,
    )
