import pandas as pd
import pytest


def assert_frame_equal(res_df, expected_df):
    """
    Returns a more legible and easy to interpret output when carrying out tests
    """

    assert res_df.to_dict(orient="records") == expected_df.to_dict(orient="records")
    pd.testing.assert_frame_equal(res_df, expected_df)


pytest.register_assert_rewrite("tests.utils")
