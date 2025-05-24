import os
from typing import Optional, Self, Type

import numpy as np
import pandas as pd
from attr import attrs
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hyper_velocity_stars_detection.data_storage import ContainerSerializerZip, StorageObjectPickle
from hyper_velocity_stars_detection.tools.noise_outliers_methods import (
    NoiseMethod,
    get_noise_method,
)


class GaussianMixtureClustering:
    """
    Clase decoradora de los métodos de clusterización de GaussianMixture
    """

    def __init__(self, **kwargs):
        self.model = GaussianMixture(**kwargs)

    def fit(self, data: pd.DataFrame):
        self.model.fit(data)
        self.labels_ = self.model.predict(data)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)


ClusterMethods = Type[DBSCAN | KMeans | HDBSCAN | GaussianMixtureClustering]
ScalersMethods = Type[StandardScaler | MinMaxScaler | None]
MAX_CLUSTER_DEFAULT = 10


class ClusterMethodsNames:
    """
    Nombres de los métodos de clustering.
    """

    DBSCAN_NAME: str = "dbscan"
    KMEANS_NAME: str = "kmeans"
    HDBSCAN_NAME: str = "hdbscan"
    GM_NAME: str = "gaussian_mixture"
    AGG_NAME: str = "agglomerative"


@attrs(auto_attribs=True)
class ClusterParamsDistribution:
    """
    Clase que engloba las distribuciones de parámetros en cada método de clustering.
    """

    params_distribution = {}

    def get_params_distribution(
        self, max_cluster: Optional[int] = None, params: Optional[dict] = None
    ) -> dict:
        """
        Método que devuelve las distribuciones de los parámetros teniendo en cuenta
        los datos de params si se les pasa.

        Parameters
        ----------
        max_cluster: Optional[int], default None
            Número máximo e clusters
        params: Optional[dict], default None
            Parámetros que pueden modificar los espacios de búsqueda.

        Returns
        -------
        params_distribution: dict
            Distribución de los parámetros.
        """
        params_distribution = self.params_distribution.copy()
        params_distribution.update(params)
        return params_distribution


class DbscanParamsDistribution(ClusterParamsDistribution):
    """
    Clase que engloba las distribuciones de parámetros en DBSCAN.
    """

    params_distribution = {
        "eps": ["unif", "eps", 0.1, 1.0],
        "min_samples": ["int", "min_samples", 3, 10],
        "metric": ["cat", "metric", ["cityblock", "euclidean", "l1", "l2", "manhattan"]],
        "algorithm": ["cat", "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]],
    }


class HdbscanParamsDistribution(ClusterParamsDistribution):
    """
    Clase que engloba las distribuciones de parámetros en HDBSCAN.
    """

    params_distribution = {
        "min_cluster_size": ["int", "min_cluster_size", 10, 100],
        "min_samples": ["int", "min_samples", 3, 10],
        "metric": ["cat", "metric", ["cityblock", "euclidean", "l1", "l2", "manhattan"]],
        "algorithm": ["cat", "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]],
    }


class KmeansParamsDistribution(ClusterParamsDistribution):
    """
    Clase que engloba las distribuciones de parámetros en Kmeans.
    """

    params_distribution = {"n_clusters": ["int", "n_clusters", 2, MAX_CLUSTER_DEFAULT]}

    def get_params_distribution(
        self, max_cluster: Optional[int] = None, params: Optional[dict] = None
    ) -> dict:
        """
        Método que devuelve las distribuciones de los parámetros teniendo en cuenta
        los datos de params si se les pasa.

        Parameters
        ----------
        max_cluster: Optional[int], default None
            Número máximo e clusters
        params: Optional[dict], default None
            Parámetros que pueden modificar los espacios de búsqueda.

        Returns
        -------
        params_distribution: dict
            Distribución de los parámetros.
        """
        if params is None:
            params = {}
        if max_cluster is None:
            max_cluster = MAX_CLUSTER_DEFAULT
        params_distribution = self.params_distribution.copy()
        params_distribution.update(params)

        if max_cluster <= 2:
            params_distribution["n_clusters"] = ["cat", "n_clusters", [2]]
            return params_distribution

        params_distribution["n_clusters"] = ["int", "n_clusters", 2, max_cluster]
        return params_distribution


class AgglomerativeClusteringParamsDistribution(ClusterParamsDistribution):
    """
    Clase que engloba las distribuciones de parámetros en Kmeans.
    """

    params_distribution = {
        "n_clusters": ["int", "n_clusters", 2, MAX_CLUSTER_DEFAULT],
        "linkage": ["cat", "linkage", ["ward", "complete", "average", "single"]],
    }

    def get_params_distribution(
        self, max_cluster: Optional[int] = None, params: Optional[dict] = None
    ) -> dict:
        """
        Método que devuelve las distribuciones de los parámetros teniendo en cuenta
        los datos de params si se les pasa.

        Parameters
        ----------
        max_cluster: Optional[int], default None
            Número máximo e clusters
        params: Optional[dict], default None
            Parámetros que pueden modificar los espacios de búsqueda.

        Returns
        -------
        params_distribution: dict
            Distribución de los parámetros.
        """
        if params is None:
            params = {}
        if max_cluster is None:
            max_cluster = MAX_CLUSTER_DEFAULT
        params_distribution = self.params_distribution.copy()
        params_distribution.update(params)

        if max_cluster <= 2:
            params_distribution["n_clusters"] = ["cat", "n_clusters", [2]]
            return params_distribution

        params_distribution["n_clusters"] = ["int", "n_clusters", 2, max_cluster]
        return params_distribution


class GaussianMixtureParamsDistribution(ClusterParamsDistribution):
    """
    Clase que engloba las distribuciones de parámetros en Kmeans.
    """

    params_distribution = {
        "n_components": ["int", "n_components", 2, MAX_CLUSTER_DEFAULT],
        "covariance_type": ["cat", "covariance_type", ["full", "tied", "diag", "spherical"]],
        "tol": ["lunif", "tol", 1e-5, 1e-3],
        "max_iter": ["int", "max_iter", 100, 300],
    }

    def get_params_distribution(
        self, max_cluster: Optional[int] = None, params: Optional[dict] = None
    ) -> dict:
        """
        Método que devuelve las distribuciones de los parámetros teniendo en cuenta
        los datos de params si se les pasa.

        Parameters
        ----------
        max_cluster: Optional[int], default None
            Número máximo e clusters
        params: Optional[dict], default None
            Parámetros que pueden modificar los espacios de búsqueda.

        Returns
        -------
        params_distribution: dict
            Distribución de los parámetros.
        """
        if params is None:
            params = {}
        if max_cluster is None:
            max_cluster = MAX_CLUSTER_DEFAULT
        params_distribution = self.params_distribution.copy()
        params_distribution.update(params)

        if max_cluster <= 2:
            params_distribution["n_components"] = ["cat", "n_components", [2]]
            return params_distribution

        params_distribution["n_components"] = ["int", "n_components", 2, max_cluster]
        return params_distribution


def get_cluster_method(method: str) -> ClusterMethods:
    """
    Función que nos devuelve el modelo de clusterizaación.

    Parameters
    ----------
    method: str
        Nombre del método a utilizar.

    Returns
    -------
    clustering: ClusterMethods
        Modelo de clusterización
    """

    methods = {
        ClusterMethodsNames.DBSCAN_NAME: DBSCAN,
        ClusterMethodsNames.KMEANS_NAME: KMeans,
        ClusterMethodsNames.HDBSCAN_NAME: HDBSCAN,
        ClusterMethodsNames.GM_NAME: GaussianMixtureClustering,
        ClusterMethodsNames.AGG_NAME: AgglomerativeClustering,
    }
    try:
        return methods[method]
    except KeyError:
        raise ValueError(f"No existe el método {method}, prueba con : {list(methods.keys())}")


def get_scaler_method(method: str | None, params: Optional[dict] = None) -> ScalersMethods:
    """
    Función que nos devuelve el método de scaler.

    Parameters
    ----------
    method: str | None
        Nombre del método a utilizar. Si es None devuelve None.
    params: Optional[dict] = None
        Parámetros del scaler.

    Returns
    -------
    scaler: ScalersMethods
        Modelo de clusterización
    """
    if method is None:
        return None
    if params is None:
        params = {}

    methods = {"minmax": MinMaxScaler, "standard": StandardScaler}
    try:
        return methods[method](**params)
    except KeyError:
        raise ValueError(f"No existe el método {method}, prueba con : {list(methods.keys())}")


def get_default_params_distribution(
    method: str, max_cluster: int = None, params_to_opt: Optional[dict] = None
) -> dict:
    """
    Función que nos devuelve la distribución por defecto de los parámetros.

    Parameters
    ----------
    method: str
        Nombre del método a utilizar.
    params_to_opt: Optional[dict] = None
        Parámetros utilizados en la optimización que pueden condicionar la distribución
        de los parámetros.

    Returns
    -------
    params_distribucion: dict
        Distribución de los parámetros por defecto
    """

    methods = {
        ClusterMethodsNames.DBSCAN_NAME: DbscanParamsDistribution(),
        ClusterMethodsNames.KMEANS_NAME: KmeansParamsDistribution(),
        ClusterMethodsNames.HDBSCAN_NAME: HdbscanParamsDistribution(),
        ClusterMethodsNames.GM_NAME: GaussianMixtureParamsDistribution(),
        ClusterMethodsNames.AGG_NAME: AgglomerativeClusteringParamsDistribution(),
    }
    try:
        return methods[method].get_params_distribution(max_cluster, params_to_opt)
    except KeyError:
        raise ValueError(f"No existe el método {method}, prueba con : {list(methods.keys())}")


def get_distance_from_references(
    labels: np.ndarray,
    cluster_data: pd.DataFrame,
    reference_cluster: pd.Series,
    columns_cluster: list[str],
) -> np.ndarray:
    """
    Función que calcula las distancias para cada uno de los cluster con los datos de referencia.

    Parameters
    ----------
    labels: np.ndarray,
        Etiqueta de cada elemento del cluster data.
    cluster_data: pd.DataFrame,
        Datos de las estrellas analizados.
    reference_cluster: pd.DataFrame
        Datos de referencia del cluster

    Returns
    -------
    distances: np.ndarray
        Distancias para cada cluster encontrado.

    """
    mean_cluster_dr2 = reference_cluster[columns_cluster].iloc[0, :].values
    unique_labels = np.unique(labels)
    distances = np.zeros(unique_labels.size)
    for i, label in enumerate(unique_labels):
        gc = cluster_data.loc[labels == label]
        mean_cluster = gc[columns_cluster].mean().values
        distances[i] = np.sqrt(np.linalg.norm(mean_cluster_dr2 - mean_cluster))
    return distances


@attrs(auto_attribs=True)
class ClusteringDetection:
    model: ClusterMethods
    scaler: ScalersMethods = None
    noise_method: NoiseMethod | None = None
    labels_: np.ndarray = None

    _storage = ContainerSerializerZip(
        serializers={
            "model": StorageObjectPickle(),
            "noise_method": StorageObjectPickle(),
            "scaler": StorageObjectPickle(),
        }
    )

    def fit(
        self,
        x: np.ndarray,
        scaler_params: Optional[dict] = None,
        noise_params: Optional[dict] = None,
        cluster_params: Optional[dict] = None,
    ):
        """
        Método que entrena el scaler, método de eliminación de outliers y clusterización
        asociados.

        Parameters
        ----------
        x: np.ndarray
            Datos a clusterizar.
        scaler_params:Optional[dict] = None
            Parámetros asociados al scaler. Por defecto es None
        noise_params: Optional[dict] = None,
            Parámetros asociados a la eliminación de outlier. Por defecto es None
        cluster_params: Optional[dict] = None,
            Parámetros asociados al método de clusterización.
        """
        if scaler_params is None:
            scaler_params = {}
        if noise_params is None:
            noise_params = {}
        if cluster_params is None:
            cluster_params = {}
        x_to_fit = x.copy()
        if self.scaler is not None:
            x_to_fit = self.scaler.fit_transform(x, **scaler_params)
        if self.noise_method is not None:
            mask = self.noise_method.fit_predict(x_to_fit, **noise_params)
            x_to_fit = x_to_fit[mask > -1]
        self.model.fit(x_to_fit, **cluster_params)
        _ = self.predict(x)

    def preprocessing(self, x: np.ndarray):
        """
        Método que preprocesa usando el scaler y método de eliminación de outliers.
        asociados.

        Parameters
        ----------
        x: np.ndarray
            Datos a clusterizar.

        Returns
        -------
        x_pre: np.ndarray
            Preprocesado de los datos.
        """
        x_pre = x.copy()
        if self.scaler is not None:
            x_pre = self.scaler.transform(x_pre)
        if self.noise_method is not None:
            mask = self.noise_method.predict(x_pre)
            x_pre = x_pre[mask > -1]
        return x_pre

    def get_main_cluster(
        self,
        cluster_data: Optional[pd.DataFrame] = None,
        reference_cluster: Optional[pd.Series] = None,
    ) -> int:
        """
        Función que calcula el cluster mayoritario

        Parameters
        ----------
        cluster_data:  Optional[pd.DataFrame] = None
            Datos de las estrellas analizados.
        reference_cluster:  Optional[pd.Series] = None
            Datos de referencia del cluster

        Returns
        -------
        label_cluster: int
            Valor de la etiqueta del cluster con mayor volumen.
        """
        labels = self.labels_
        unique_lab = np.unique(labels[labels > -1])
        if unique_lab.size == 0:
            return -1
        j = np.argmax(np.bincount(labels[labels > -1]))
        if isinstance(reference_cluster, pd.Series) and isinstance(cluster_data, pd.DataFrame):
            distances = get_distance_from_references(
                labels=labels[labels > -1],
                cluster_data=self.preprocessing(cluster_data),
                reference_cluster=reference_cluster,
                columns_cluster=cluster_data.columns,
            )
            j = np.argmin(distances)
        return int(unique_lab[j])

    def save(self, path: str):
        """
        Método que guarda los reusltados en un archivo zip.

        Parameters
        ----------
        path: str
            Ruta donde se quiere guardar los archivos.
        """
        path_name = os.path.join(path, "clustering_method")
        container = {"model": self.model, "noise": self.noise_method, "scaler": self.scaler}
        self._storage.save(path_name, container)

    def predict(self, x: np.ndarray):
        """
        Método que predice usando el scaler, método de eliminación de outliers y clusterización
        asociados.

        Parameters
        ----------
        x: np.ndarray
            Datos a clusterizar.

        Returns
        -------
        prediction: np.ndarray
            Predicción de la clusterización.
        """
        x_pred = x.copy()
        self.labels_ = np.ones(x_pred.shape[0])
        if self.scaler is not None:
            x_pred = self.scaler.transform(x_pred)
        if self.noise_method is not None:
            mask = self.noise_method.predict(x_pred)
            self.labels_[mask > -1] = self.model.predict(x_pred[mask > -1])
            self.labels_[mask == -1] = -1
            return self.labels_

        self.labels_ = self.model.predict(x_pred)
        return self.labels_

    @classmethod
    def from_cluster_params(
        cls,
        cluster_method: str,
        cluster_params: dict,
        scaler_method: Optional[str] = None,
        scaler_params: Optional[dict] = None,
        noise_method: Optional[str] = None,
        noise_params: Optional[dict] = None,
    ) -> Self:
        clustering = get_cluster_method(cluster_method)(**cluster_params)
        scaler = get_scaler_method(scaler_method, scaler_params)
        noise = get_noise_method(noise_method, noise_params)
        return cls(clustering, scaler, noise)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Método que carga los resultados de la clusterización.

        Parameters
        ----------
        path: str
            Ruta donde se quiere guardar

        Returns
        -------
        object: Self
            Objeto instanciado.
        """
        path_name = os.path.join(path, "clustering_method.zip")
        params = cls._storage.load(path_name)
        return cls(**params)
