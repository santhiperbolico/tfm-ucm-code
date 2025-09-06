import os.path
from tempfile import TemporaryDirectory
from typing import Self
from unittest.mock import patch

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from attr import attrs
from pandas.testing import assert_frame_equal

from hyper_velocity_stars_detection.sources.catalogs import Chandra, XMMNewton, XRSCatalog
from hyper_velocity_stars_detection.sources.source import AstroObject
from hyper_velocity_stars_detection.sources.xray_source import XRSourceData, get_xrs_catalog


@attrs(auto_attribs=True)
class GlobularCluster:
    name: str
    coords: SkyCoord
    radius: float

    @classmethod
    def load_cluster(cls, name: str, ra: float, dec: float, radius: float) -> Self:
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        return cls(name, coords, radius)


@pytest.fixture
def cluster():
    return GlobularCluster.load_cluster("ngc 104", 6.02363, -72.08128, 43 / 60)


@pytest.fixture
@patch("hyper_velocity_stars_detection.sources.utils.Simbad", autospec=True)
def astro_object(simbad_class_mock, cluster):
    simbad_mock = simbad_class_mock.return_value
    simbad_mock.query_object.return_value = Table.read("tests/test_data/result_simbad.txt")
    astro_object = AstroObject.get_object(cluster.name)
    return astro_object


def query_region_side_effect(coords, mission, radius, fields) -> Table:
    return Table.read(f"tests/test_data/{mission}_heasarc.fits")


@pytest.mark.parametrize("catalog", [XMMNewton, Chandra, [XMMNewton, Chandra]])
def test_get_xrs_catalog(catalog):
    if isinstance(catalog, list):
        catalog_name = [cat.catalog_name for cat in catalog]
        expected = [cat for cat in catalog]
    else:
        catalog_name = catalog.catalog_name
        expected = [catalog]
    result = get_xrs_catalog(catalog_name=catalog_name)
    assert isinstance(result, list)
    for i in range(len(result)):
        assert isinstance(result[i], expected[i])


@pytest.mark.parametrize("catalog", [XMMNewton(), Chandra()])
@patch("hyper_velocity_stars_detection.sources.catalogs.Heasarc", autospec=True)
def test_xrsource_data_load_astro_data(heasarc_class_mock, catalog, astro_object):
    heasarc_class_mock.query_region.return_value = Table.read(
        f"tests/test_data/{catalog.catalog_name}_heasarc.fits"
    )
    catalog_name = None
    if isinstance(catalog, list):
        catalog_name = [cat.catalog_name for cat in catalog]
    if isinstance(catalog, XRSCatalog):
        catalog_name = catalog.catalog_name
    xrs_data = XRSourceData.load_data(astro_object.name, catalog_name=catalog_name, radius_scale=1)
    assert isinstance(xrs_data, XRSourceData)
    assert isinstance(xrs_data.data, pd.DataFrame)
    assert not xrs_data.data.empty


@pytest.mark.parametrize("catalog", [None, [XMMNewton(), Chandra()]])
@patch("hyper_velocity_stars_detection.sources.catalogs.Chandra.download_data", autospec=True)
@patch("hyper_velocity_stars_detection.sources.catalogs.XMMNewton.download_data", autospec=True)
def test_xrsource_data_load_astro_data_multi(xmmn_dwn_mock, chnd_dwn_mock, catalog, astro_object):
    df_xmmn = pd.read_csv("tests/test_data/xmmnewton_data.csv")
    df_chnd = pd.read_csv("tests/test_data/chandra_data.csv")
    xmmn_dwn_mock.return_value = df_xmmn
    chnd_dwn_mock.return_value = df_chnd
    expected = pd.concat((df_xmmn, df_chnd)).reset_index(drop=True)
    catalog_name = None
    if isinstance(catalog, list):
        catalog_name = [cat.catalog_name for cat in catalog]
    if isinstance(catalog, XRSCatalog):
        catalog_name = catalog.catalog_name
    xrs_data = XRSourceData.load_data(astro_object.name, catalog_name=catalog_name, radius_scale=1)
    assert isinstance(xrs_data, XRSourceData)
    assert_frame_equal(xrs_data.data, expected)


@patch("hyper_velocity_stars_detection.sources.catalogs.Chandra.download_data", autospec=True)
@patch("hyper_velocity_stars_detection.sources.catalogs.XMMNewton.download_data", autospec=True)
def test_xrsource_data_save_load_from_zip(xmmn_dwn_mock, chnd_dwn_mock, astro_object):
    df_xmmn = pd.read_csv("tests/test_data/xmmnewton_data.csv")
    df_chnd = pd.read_csv("tests/test_data/chandra_data.csv")
    xmmn_dwn_mock.return_value = df_xmmn
    chnd_dwn_mock.return_value = df_chnd
    astro_data = XRSourceData.load_data(astro_object.name, radius_scale=1)
    with TemporaryDirectory() as tmp_dir:
        filename = "xrsource_" + astro_data.data_name
        file = os.path.join(tmp_dir, filename)
        astro_data.save(tmp_dir)
        new_data = XRSourceData.load(file)

    assert_frame_equal(astro_data.data, new_data.data)
    assert new_data.data_name == astro_data.data_name


@patch("hyper_velocity_stars_detection.sources.catalogs.Chandra.download_data", autospec=True)
@patch("hyper_velocity_stars_detection.sources.catalogs.XMMNewton.download_data", autospec=True)
def test_xrsource_data_get_data(xmmn_dwn_mock, chnd_dwn_mock, astro_object):
    df_xmmn = pd.read_csv("tests/test_data/xmmnewton_data.csv")
    df_chnd = pd.read_csv("tests/test_data/chandra_data.csv")
    xmmn_dwn_mock.return_value = df_xmmn
    chnd_dwn_mock.return_value = df_chnd
    astro_data = XRSourceData.load_data(astro_object.name, radius_scale=1)
    result_xmmn = astro_data.get_data("xmmnewton")
    result_chnd = astro_data.get_data("chandra")
    assert_frame_equal(df_xmmn, result_xmmn, check_dtype=False)
    assert_frame_equal(df_chnd, result_chnd, check_dtype=False)
