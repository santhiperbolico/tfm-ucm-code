from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from hyper_velocity_stars_detection.cluster_detection.cluster_detection import (
    ClusteringDetection,
    ClusteringResults,
)


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


def test_clusteringresults_save_load(data_labels):
    with TemporaryDirectory() as temp_dir:
        data_labels.save(temp_dir)
        new_data = ClusteringResults.load(temp_dir)

    pd.testing.assert_frame_equal(new_data.df_stars, data_labels.df_stars)
    assert new_data.columns == data_labels.columns
    assert (new_data.labels == data_labels.labels).all()
