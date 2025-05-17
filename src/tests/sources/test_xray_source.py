import os.path
from tempfile import TemporaryDirectory
from typing import Self
from unittest.mock import patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from attr import attrs

from hyper_velocity_stars_detection.sources.xray_source import (
    XCatalog,
    XCatalogParams,
    XSource,
    get_main_id,
    get_obs_id,
)


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


def query_region_side_effect(coords, mission, radius, fields) -> Table:
    return Table.read(f"tests/test_data/{mission}_heasarc.fits")


@pytest.mark.parametrize(
    "mission, columns, expected_shape",
    [
        (XCatalogParams.XMNNEWTON[0], XCatalogParams.XMNNEWTON[1], (1, 13)),
        (XCatalogParams.CHANDRA[0], XCatalogParams.CHANDRA[1], (36, 13)),
    ],
)
@patch("hyper_velocity_stars_detection.sources.xray_source.Heasarc", autospec=True)
def test_xcatalog_download_data(heasarc_class_mock, mission, columns, expected_shape, cluster):
    heasarc_class_mock.query_region.return_value = Table.read(
        f"tests/test_data/{mission}_heasarc.fits"
    )
    catalog = XCatalog(mission, columns)
    result = catalog.download_data(cluster.coords, cluster.radius)
    n_cols = len(XCatalog.format_columns)
    assert result.shape == expected_shape
    assert result.columns.isin(list(XCatalog.format_columns.keys())).sum() == n_cols


@pytest.mark.parametrize("name, expected", [("NGC 104", "NGC_104"), ("47 TUC", "NGC_104")])
@patch("hyper_velocity_stars_detection.sources.xray_source.Simbad", autospec=True)
def test_get_main_id(simbad_class_mock, name, expected, cluster):
    simbad_class_mock.query_object.return_value = Table.read("tests/test_data/result_simbad.txt")
    result = get_main_id(name)
    assert result == expected


def test_get_obs_id():
    obs_list = [123456789, 291, "0123456789"]
    expected_list = ["0123456789", "0000000291", "0123456789"]
    result = get_obs_id(obs_list)
    assert result == expected_list


@patch("hyper_velocity_stars_detection.sources.xray_source.Heasarc", autospec=True)
def test_xsource(heasarc_class_mock, cluster):
    heasarc_class_mock.query_region.side_effect = query_region_side_effect
    with TemporaryDirectory() as temp_dir:
        xsource = XSource(temp_dir)
        xsource.download_data(cluster.coords, cluster.radius)
    assert xsource.results.shape == (37, 13)


@patch("hyper_velocity_stars_detection.sources.xray_source.Heasarc", autospec=True)
def test_xsource_save_load(heasarc_class_mock, cluster):
    heasarc_class_mock.query_region.side_effect = query_region_side_effect
    with TemporaryDirectory() as temp_dir:
        xsource = XSource(temp_dir)
        xsource.download_data(cluster.coords, cluster.radius)
        xsource.save()
        xsource_new = XSource(temp_dir)
        xsource_new.load()
        assert os.path.exists(os.path.join(temp_dir, "xsource.zip"))
    assert xsource.results.shape == (37, 13)


@patch("hyper_velocity_stars_detection.sources.xray_source.Heasarc", autospec=True)
def test_xsource_save_load_from_path(heasarc_class_mock, cluster):
    heasarc_class_mock.query_region.side_effect = query_region_side_effect
    with TemporaryDirectory() as temp_dir:
        xsource = XSource(temp_dir)
        xsource.download_data(cluster.coords, cluster.radius)
        xsource.save(temp_dir)
        xsource_new = XSource(temp_dir)
        xsource_new.load(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "xsource.zip"))
    assert xsource.results.shape == (37, 13)
