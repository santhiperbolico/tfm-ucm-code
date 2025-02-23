import os.path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.table.table import Table
from attr import attrs

from high_velocity_stars_detection.astrobjects import (
    AstroObject,
    AstroObjectData,
    AstroObjectProject,
)
from high_velocity_stars_detection.etls.catalogs import CatalogsType


@attrs(auto_attribs=True)
class GlobularCluster:
    name: str
    ra: float
    dec: float


@pytest.fixture
def cluster():
    return GlobularCluster("ngc 104", 6.02363, -72.08128)


@pytest.fixture
@patch("high_velocity_stars_detection.etls.download_data.Heasarc", autospec=True)
def astro_object(heasarc_class_mock, cluster):
    heasarc_mock = heasarc_class_mock.return_value
    heasarc_mock.query_object.return_value = Table.read("tests/test_data/result_heasarc.fits")
    astro_object = AstroObject.get_object(cluster.name)
    return astro_object


def test_astroobject_get_object(astro_object, cluster):
    assert isinstance(astro_object, AstroObject)
    assert astro_object.coord.ra.value == pytest.approx(cluster.ra)
    assert astro_object.coord.dec.value == pytest.approx(cluster.dec)


@pytest.mark.parametrize("radius_type", ["core_radius", "half_light_radius", "vision_fold_radius"])
@patch("high_velocity_stars_detection.etls.catalogs.Gaia", autospec=True)
@patch("astroquery.utils.tap.model.job.Job", autospec=True)
def test_astroobject_download_object(
    job_class_mock, gaia_class_mock, radius_type, astro_object, cluster
):
    job_mock = job_class_mock.return_value
    job_mock.get_results.return_value = Table.read("tests/test_data/result_gaia.fits")
    gaia_class_mock.launch_job_async.return_value = job_mock

    astro_object = AstroObject.get_object(cluster.name)
    df_result = astro_object.download_object(
        CatalogsType.GAIA_DR3, radius_scale=1, radius_type=radius_type
    )
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == 50


@pytest.mark.parametrize(
    "file_to_read, catalog_name, radius_scale",
    [
        ("data_file.vot", None, None),
        (None, CatalogsType.GAIA_DR3, 1.0),
    ],
)
@patch("high_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_astroobject_read_object(
    table_class_mock, file_to_read, catalog_name, radius_scale, astro_object
):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    df_result = astro_object.read_object("path_dir", file_to_read, catalog_name, radius_scale)
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape[0] == 50


@patch("high_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_astroobject_copy_set_extra_metrics(table_class_mock, astro_object):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    _ = astro_object.read_object("path_dir", "file.vot")
    df_data = astro_object.copy_set_extra_metrics()
    columns = ["pmra_kms", "pmdec_kms", "pm_kms", "pm", "pm_l", "pm_b", "uwe", "ruwe"]
    assert isinstance(df_data, pd.DataFrame)
    assert np.array(col in df_data.columns for col in columns).all()


@patch("high_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_astroobject_qualify_data(table_class_mock, astro_object):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    _ = astro_object.read_object("path_dir", "file.vot")
    result = astro_object.qualify_data()
    columns = ["pmra_kms", "pmdec_kms", "pm_kms", "pm", "pm_l", "pm_b", "uwe", "ruwe"]
    for df_data in result:
        assert isinstance(df_data, pd.DataFrame)
        assert np.array(col in df_data.columns for col in columns).all()


@patch("high_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_astroobject_data_load_from_object(table_class_mock, astro_object):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    _ = astro_object.read_object("path_dir", "file.vot")
    data = AstroObjectData.load_data_from_object(astro_object, radio_scale=1)
    columns = ["pmra_kms", "pmdec_kms", "pm_kms", "pm", "pm_l", "pm_b", "uwe", "ruwe"]
    for key, df_data in data.data.items():
        assert isinstance(df_data, pd.DataFrame)
        assert np.array(col in df_data.columns for col in columns).all()

    assert len(data.data) == 4


@patch("high_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_astroobject_data_save_load_from_zip(table_class_mock, astro_object):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    _ = astro_object.read_object("path_dir", "file.vot")
    data = AstroObjectData.load_data_from_object(astro_object, radio_scale=1)
    with TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, data.data_name) + ".zip"
        data.save_data(tmp_dir)
        new_data = AstroObjectData.load_object_from_zip(file)

    assert len(new_data.data) == 4
    assert new_data.data_name == data.data_name


@patch("high_velocity_stars_detection.etls.catalogs.Table", autospec=True)
def test_astroobject_project_save_load_from_zip(table_class_mock, astro_object):
    table_class_mock.read.return_value = Table.read("tests/test_data/result_gaia.fits")
    _ = astro_object.read_object("path_dir", "file.vot")
    data_list = [
        AstroObjectData.load_data_from_object(astro_object, radio_scale=1),
        AstroObjectData.load_data_from_object(astro_object, radio_scale=2),
    ]
    project = AstroObjectProject("ngc 104", data_list)
    with TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, project.name)
        project.save_project(tmp_dir)
        new_project = AstroObjectProject.load_project(path)

    assert len(new_project.data_list) == 2
    assert new_project.name == project.name
