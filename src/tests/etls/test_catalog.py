import os.path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest
from astropy.table.table import Table
from attr import attrs

from hyper_velocity_stars_detection.etls.catalogs import Catalog, CatalogsTables, CatalogsType


@attrs(auto_attribs=True)
class GlobularCluster:
    name: str
    ra: float
    dec: float


@pytest.fixture
def cluster():
    return GlobularCluster("ngc 104", 6.02363, -72.08128)


@pytest.mark.parametrize(
    "catalog_name, catalog_table",
    [
        (CatalogsType.GAIA_DR2, CatalogsTables.GAIA_DR2),
        (CatalogsType.GAIA_DR3, CatalogsTables.GAIA_DR3),
    ],
)
def test_catalog_get_catalog(catalog_name, catalog_table):
    catalog = Catalog.get_catalog(catalog_name)
    assert catalog.catalog_table == catalog_table


@patch("hyper_velocity_stars_detection.etls.catalogs.Gaia", autospec=True)
@patch("astroquery.utils.tap.model.job.Job", autospec=True)
def test_catalog_download(job_class_mock, gaia_class_mock, cluster):
    job_mock = job_class_mock.return_value
    job_mock.get_results.return_value = Table.read("tests/test_data/result_gaia.fits")
    gaia_class_mock.launch_job_async.return_value = job_mock

    catalog = Catalog(CatalogsTables.GAIA_DR3)
    with TemporaryDirectory() as tmp_dir:
        output_file = os.path.join(tmp_dir, "test_data.vot")
        df_result = catalog.download_results(cluster.ra, cluster.dec, 1, output_file)

    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == 50


@patch("hyper_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_catalog_read(table_class_mock, cluster):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    df_result = Catalog.read_catalog("data_file.vot")
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == 50
