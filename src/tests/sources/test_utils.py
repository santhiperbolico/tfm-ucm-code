from unittest.mock import patch

import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from attr import attrs

from hyper_velocity_stars_detection.sources.utils import (
    fix_parallax,
    get_object_from_heasarc,
    get_object_from_simbad,
    get_skycoords,
)


@attrs(auto_attribs=True)
class GlobularCluster:
    name: str
    ra: float
    dec: float


@pytest.fixture
def cluster():
    return GlobularCluster("ngc 104", 6.02363, -72.08128)


@pytest.fixture()
def df_data():
    return pd.read_csv("tests/test_data/df_data.csv")


@patch("hyper_velocity_stars_detection.sources.utils.Heasarc", autospec=True)
def test_get_object_from_heasarc(heasarc_class_mock, cluster):
    heasarc_mock = heasarc_class_mock.return_value
    heasarc_mock.query_object.return_value = Table.read("tests/test_data/result_heasarc.fits")
    result = get_object_from_heasarc(cluster)
    assert isinstance(result, Table)
    assert result["RA"] == pytest.approx(cluster.ra, abs=1e-3)
    assert result["DEC"] == pytest.approx(cluster.dec, abs=1e-3)


@patch("hyper_velocity_stars_detection.sources.utils.Simbad", autospec=True)
def test_get_object_from_simbad(simbad_class_mock, cluster):
    simbad_mock = simbad_class_mock.return_value
    simbad_mock.query_object.return_value = Table.read("tests/test_data/result_simbad.txt")
    result = get_object_from_simbad(cluster.name)
    assert isinstance(result, Table)
    assert result["RA"] == pytest.approx(cluster.ra, abs=1e-2)
    assert result["DEC"] == pytest.approx(cluster.dec, abs=1e-2)


@patch("hyper_velocity_stars_detection.sources.utils.Simbad", autospec=True)
def test_get_skycoords(simbad_class_mock, cluster):
    simbad_mock = simbad_class_mock.return_value
    simbad_mock.query_object.return_value = Table.read("tests/test_data/result_simbad.txt")
    result = get_object_from_simbad(cluster.name)
    coords = get_skycoords(result, u.deg, u.deg)
    assert isinstance(coords, SkyCoord)
    assert cluster.ra == pytest.approx(cluster.ra, abs=1e-2)
    assert cluster.dec == pytest.approx(cluster.dec, abs=1e-2)


def test_fix_parallax(df_data):
    df_data = fix_parallax(df_data)
    assert "parallax_corrected" in df_data.columns
    assert (df_data.parallax_corrected > df_data.parallax).all()
