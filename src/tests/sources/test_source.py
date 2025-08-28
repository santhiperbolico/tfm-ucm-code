import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest
from astropy.table.table import Table
from attr import attrs
from pandas.testing import assert_frame_equal

from hyper_velocity_stars_detection.sources.catalogs import GaiaDR3
from hyper_velocity_stars_detection.sources.source import AstroMetricData, AstroObject


@attrs(auto_attribs=True)
class GlobularCluster:
    name: str
    ra: float
    dec: float


@pytest.fixture
def cluster():
    return GlobularCluster("ngc 104", 6.02363, -72.08128)


@pytest.fixture
@patch("hyper_velocity_stars_detection.sources.utils.Simbad", autospec=True)
def astro_object(simbad_class_mock, cluster):
    simbad_mock = simbad_class_mock.return_value
    simbad_mock.query_object.return_value = Table.read("tests/test_data/result_simbad.txt")
    astro_object = AstroObject.get_object(cluster.name)
    return astro_object


def test_astroobject_get_object(astro_object, cluster):
    assert isinstance(astro_object, AstroObject)
    assert astro_object.coord.ra.value == pytest.approx(cluster.ra, abs=1e-2)
    assert astro_object.coord.dec.value == pytest.approx(cluster.dec, abs=1e-2)


def test_astroobject_load_save_object(astro_object):
    with TemporaryDirectory() as temp_dir:
        astro_object.save(temp_dir)
        file = "astro_object_" + astro_object.main_id
        file_path = os.path.join(temp_dir, file)
        new_object = AstroObject.load(file_path)

    assert isinstance(new_object, AstroObject)
    assert new_object.coord == astro_object.coord
    assert new_object.name == astro_object.name
    assert new_object.main_id == astro_object.main_id
    assert_frame_equal(new_object.info.to_pandas(), astro_object.info.to_pandas())


@patch("hyper_velocity_stars_detection.sources.catalogs.GaiaDR3._download_data", autospec=True)
def test_astroobject_data_load_astro_data(gaia_download_data_mock, astro_object):
    gaia_download_data_mock.return_value = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData.load_data(astro_object.name, GaiaDR3.catalog_name, radius_scale=1)
    columns = ["pmra_kms", "pmdec_kms", "pm_kms", "pm", "pm_l", "pm_b", "ruwe", "parallax_orig"]
    assert isinstance(astro_data.data, pd.DataFrame)
    assert astro_data.data.columns.isin(columns).sum() == len(columns)


@patch("hyper_velocity_stars_detection.sources.catalogs.GaiaDR3._download_data", autospec=True)
def test_astroobject_data_save_load_from_zip(gaia_download_data_mock, astro_object):
    gaia_download_data_mock.return_value = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData.load_data(astro_object.name, GaiaDR3.catalog_name, radius_scale=1)
    with TemporaryDirectory() as tmp_dir:
        filename = "astro_data_" + astro_data.data_name
        file = os.path.join(tmp_dir, filename)
        astro_data.save(tmp_dir)
        new_data = AstroMetricData.load(file)

    assert_frame_equal(astro_data.data, new_data.data)
    assert new_data.data_name == astro_data.data_name


def test_astroobject_data_fix_parallax(astro_object):
    data = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData(astro_object, GaiaDR3(), radio_scale=1.0, data=data.copy())
    assert_frame_equal(astro_data.data, data)
    astro_data.fix_parallax()
    assert (astro_data.data.parallax > astro_data.data.parallax_orig).all()


def test_astroobject_data_calculate_pm_kms(astro_object):
    data = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData(astro_object, GaiaDR3(), radio_scale=1.0, data=data.copy())
    assert_frame_equal(astro_data.data, data)
    astro_data.calculate_pm_to_kms()

    columns = ["pmra_kms", "pmdec_kms", "pm_kms", "pm", "pm_l", "pm_b"]
    assert astro_data.data.columns.isin(columns).sum() == len(columns)


@pytest.mark.parametrize(
    "sample_type, n_rows",
    [
        ("df_c1", 10),
        ("df_c2", 0),
        ("df_c3", 0),
        ("df_c4", 1),
    ],
)
def test_astroobject_data_get_data(sample_type, n_rows, astro_object):
    data = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData(astro_object, GaiaDR3(), radio_scale=1.0, data=data.copy())

    df_c = astro_data.get_data(sample_type)
    assert df_c.shape[0] == n_rows


def test_astroobject_data_get_data_error(astro_object):
    data = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData(astro_object, GaiaDR3(), radio_scale=1.0, data=data.copy())

    with pytest.raises(ValueError):
        astro_data.get_data("df_data")
