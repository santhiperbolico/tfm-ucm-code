from typing import Optional, Type

import pandas as pd
from attr import attrs
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


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
    }


class HdbscanParamsDistribution(ClusterParamsDistribution):
    """
    Clase que engloba las distribuciones de parámetros en HDBSCAN.
    """

    params_distribution = {
        "min_cluster_size": ["int", "min_cluster_size", 10, 100],
        "min_samples": ["int", "min_samples", 3, 10],
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
