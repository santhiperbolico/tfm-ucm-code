import os.path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from attr import attrs
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler

from hyper_velocity_stars_detection.cluster_detection.cluster_detection import (
    ClusteringDetection,
    ClusteringResults,
)
from hyper_velocity_stars_detection.globular_clusters import GlobularClusterAnalysis
from hyper_velocity_stars_detection.sources.source import AstroMetricData
from hyper_velocity_stars_detection.sources.xray_source import XRSourceData
from hyper_velocity_stars_detection.variables_names import GLOBULAR_CLUSTER_ANALYSIS


@attrs(auto_attribs=True)
class GlobularCluster:
    name: str
    ra: float
    dec: float


@pytest.fixture
def cluster():
    return GlobularCluster("ngc 104", 6.02363, -72.08128)


@pytest.fixture
def data_labels() -> ClusteringResults:
    np.random.seed(1234)
    mean_cluster = np.array([[5.1, -2.1, 0.20], [3.1, 2.0, 0.10], [1.1, -7.0, 0.40]])
    std_cluster = np.array(
        [
            [0.20, 1.0, 0.06],
            [0.10, 1.3, 0.09],
            [1.0, 0.7, 0.03],
        ]
    )
    columns = ["pmra", "pmdec", "parallax"]
    size_cluster = np.array([1000, 500, 200])
    data = np.zeros((int(size_cluster.sum()), mean_cluster.shape[1]))
    labels = np.zeros(data.shape[0]).astype(int)
    item_0 = 0
    label = 0
    for size in size_cluster:
        data[item_0 : item_0 + size] = np.random.normal(
            loc=mean_cluster[label, :],
            scale=std_cluster[label, :],
            size=(size, mean_cluster.shape[1]),
        )
        labels[item_0 : item_0 + size] = label
        label += 1
        item_0 = size

    df_data = pd.DataFrame(data=StandardScaler().fit_transform(data), columns=columns)
    clustering = ClusteringDetection.from_cluster_params("dbscan", {}, "standard")
    clustering.labels_ = labels
    return ClusteringResults(df_data, columns, columns, clustering)


@pytest.fixture
@patch("hyper_velocity_stars_detection.sources.catalogs.GaiaDR3._download_data", autospec=True)
def astrometric_data(gaia_download_data_mock, cluster):
    gaia_download_data_mock.return_value = pd.read_csv("tests/test_data/df_data.csv")
    astro_data = AstroMetricData.load_data(cluster.name, "gaiadr3", radius_scale=1)
    return astro_data


@pytest.fixture
@patch("hyper_velocity_stars_detection.sources.catalogs.Chandra.download_data", autospec=True)
@patch("hyper_velocity_stars_detection.sources.catalogs.XMMNewton.download_data", autospec=True)
def xrsource(xmmn_dwn_mock, chnd_dwn_mock, cluster):
    df_xmmn = pd.read_csv("tests/test_data/xmmnewton_data.csv")
    df_chnd = pd.read_csv("tests/test_data/chandra_data.csv")
    xmmn_dwn_mock.return_value = df_xmmn
    chnd_dwn_mock.return_value = df_chnd
    xrsource = XRSourceData.load_data(cluster.name, radius_scale=1)
    return xrsource


def test_astroobject_project_save_load(xrsource, astrometric_data):
    gc_data = GlobularClusterAnalysis(astrometric_data, xrsource)
    with TemporaryDirectory() as tmp_dir:
        gc_data.save(tmp_dir)
        file = f"{GLOBULAR_CLUSTER_ANALYSIS}_{gc_data.name}"
        new_gc = GlobularClusterAnalysis.load(os.path.join(tmp_dir, file))

    assert_frame_equal(new_gc.astro_data.data, astrometric_data.data)
    assert_frame_equal(new_gc.xrsource.data, xrsource.data)


def test_astroobject_project_save_load_clustering(xrsource, astrometric_data, data_labels):
    gc_data = GlobularClusterAnalysis(astrometric_data, xrsource, data_labels)
    with TemporaryDirectory() as tmp_dir:
        gc_data.save(tmp_dir)
        file = f"{GLOBULAR_CLUSTER_ANALYSIS}_{gc_data.name}"
        new_gc = GlobularClusterAnalysis.load(os.path.join(tmp_dir, file))

    assert_frame_equal(new_gc.astro_data.data, astrometric_data.data)
    assert_frame_equal(new_gc.xrsource.data, xrsource.data)
    assert_frame_equal(new_gc.clustering_results.gc, data_labels.gc)


@patch("hyper_velocity_stars_detection.sources.catalogs.GaiaDR3._download_data", autospec=True)
@patch("hyper_velocity_stars_detection.sources.catalogs.Chandra.download_data", autospec=True)
@patch("hyper_velocity_stars_detection.sources.catalogs.XMMNewton.download_data", autospec=True)
def test_astroobject_project_load_globular_cluster(
    xmmn_dwn_mock, chnd_dwn_mock, gaia_download_data_mock, cluster
):
    df_xmmn = pd.read_csv("tests/test_data/xmmnewton_data.csv")
    df_chnd = pd.read_csv("tests/test_data/chandra_data.csv")
    xmmn_dwn_mock.return_value = df_xmmn
    chnd_dwn_mock.return_value = df_chnd
    xrsource = XRSourceData.load_data(cluster.name, radius_scale=1)
    df_gaia = pd.read_csv("tests/test_data/df_data.csv")
    gaia_download_data_mock.return_value = df_gaia
    gc_data = GlobularClusterAnalysis.load_globular_cluster(
        name=cluster.name, catalog_name="gaiadr3", radius_scale=1
    )
    assert_frame_equal(gc_data.astro_data.data, df_gaia)
    assert_frame_equal(gc_data.xrsource.data, xrsource.data)
