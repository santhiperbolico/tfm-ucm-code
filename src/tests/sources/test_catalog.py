from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.table.table import Table
from attr import attrs

from hyper_velocity_stars_detection.sources.catalogs import GaiaDR2, GaiaDR3, GaiaFPR, get_catalog


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


@pytest.mark.parametrize("catalog", [GaiaDR2, GaiaDR3, GaiaFPR])
@patch("hyper_velocity_stars_detection.sources.catalogs.Gaia", autospec=True)
@patch("astroquery.utils.tap.model.job.Job", autospec=True)
def test_catalog_download(job_class_mock, gaia_class_mock, catalog, cluster):
    job_mock = job_class_mock.return_value
    job_mock.get_results.return_value = Table.read("tests/test_data/result_gaia.fits")
    gaia_class_mock.launch_job_async.return_value = job_mock

    catalog_i = catalog()
    df_result = catalog_i.download_data(cluster.ra, cluster.dec, 1)

    assert isinstance(df_result, pd.DataFrame)
    assert (df_result.shape[0] <= 50) and (df_result.shape[0] >= 49)


@pytest.mark.parametrize("catalog", [GaiaDR2, GaiaDR3, GaiaFPR])
@patch("hyper_velocity_stars_detection.sources.catalogs.StorageObjectTableVotable", autospec=True)
def test_catalog_read(storage_class_mock, catalog, cluster):
    storage_class_mock.load.return_value = Table.read("tests/test_data/result_gaia.fits")
    df_result = catalog.read_catalog("data_file.vot")
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == 50


@pytest.mark.parametrize("catalog", [GaiaDR2, GaiaDR3, GaiaFPR])
def test_get_catalog(catalog):
    result = get_catalog(catalog.catalog_name)
    assert isinstance(result, catalog)


def test_get_catalog_error():
    with pytest.raises(ValueError):
        get_catalog("error_catalog_name")


def test_fix_parallax(df_data):
    parallax_corrected = GaiaDR3().fix_parallax_zero_point(df_data)
    assert isinstance(parallax_corrected, np.ndarray)
    assert (parallax_corrected > df_data.parallax.values).all()


@pytest.mark.parametrize("catalog", [GaiaDR2, GaiaFPR])
def test_fix_parallax_error(catalog, df_data):
    with pytest.raises(NotImplementedError):
        catalog().fix_parallax_zero_point(df_data)


@pytest.mark.parametrize(
    "column",
    [
        "parallax",
        "phot_g_mean_mag",
        "nu_eff_used_in_astrometry",
        "nu_eff_used_in_astrometry",
        "pseudocolour",
        "ecl_lat",
        "astrometric_params_solved",
    ],
)
def test_fix_parallax_error_gaiadr3(column, df_data):
    df = df_data.drop(columns=column)
    with pytest.raises(ValueError):
        GaiaDR3().fix_parallax_zero_point(df)
