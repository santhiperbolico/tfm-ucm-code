import pandas as pd
import pytest

from hyper_velocity_stars_detection.etls.ruwe_tools.dr3.ruwetools import U0Interpolator


@pytest.fixture()
def df_data():
    return pd.read_csv("tests/test_data/df_data.csv")


def test_u0interpolator_get_uwe(df_data):
    u0_object = U0Interpolator(5)
    uwe = u0_object.get_uwe_from_gaia(df_data)
    assert (uwe > 0).all()
    assert uwe.size == df_data.shape[0]


def test_u0interpolator_get_u0(df_data):
    u0_object = U0Interpolator(5)
    u0 = u0_object.get_u0(df_data)
    assert (u0 > 0).all()
    assert u0.size == df_data.shape[0]


def test_u0interpolator_get_ruwe(df_data):
    u0_object = U0Interpolator(5)
    ruwe = u0_object.get_ruwe_from_gaia(df_data)
    assert (ruwe > 0).all()
    assert ruwe.size == df_data.shape[0]
