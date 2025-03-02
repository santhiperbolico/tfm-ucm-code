import logging
import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from attr import attrs
from optuna import create_study
from optuna.trial import Trial
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageObjectPandasCSV,
    StorageObjectPickle,
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


ClusterMethods = Union[DBSCAN, KMeans, HDBSCAN, GaussianMixtureClustering]

DEFAULT_COLS = ["pmra", "pmdec", "parallax"]
DEFAULT_COLS_CLUS = DEFAULT_COLS + ["bp_rp", "phot_g_mean_mag"]


class ClusterMethodsNames:
    DBSCAN_NAME = "dbscan"
    KMEANS_NAME = "kmeans"
    HDBSCAN_NAME = "hdbscan"
    GM_NAME = "gaussian_mixture"


class ClusterParamsDistribution:
    DBSCAN_PARAMS = {
        "eps": ["unif", "eps", 0.1, 1.0],
        "min_samples": ["int", "min_samples", 3, 10],
    }
    HDBSCAN_PARAMS = {
        "min_cluster_size": ["int", "min_cluster_size", 10, 100],
        "min_samples": ["int", "min_samples", 3, 10],
    }
    KMEANS_PARAMS = {"n_clusters": ["int", "n_clusters", 2, 10]}
    GM_PARAMS = {
        "n_components": ["int", "n_components", 2, 10],
        "covariance_type": ["cat", "covariance_type", ["full", "tied", "diag", "spherical"]],
        "tol": ["lunif", "tol", 1e-5, 1e-3],
        "max_iter": ["int", "max_iter", 100, 300],
    }


class DistributionTrialDontExist(Exception):
    """
    Error que indica que no existe una distribucion.
    """

    pass


def get_main_cluster(labels: np.ndarray) -> int:
    """
    Función que calcula el cluster mayoritario
    """
    unique_lab = np.unique(labels[labels > -1])
    j = np.argmax(np.bincount(labels[labels > -1]))
    return unique_lab[j]


def score_cluster(df: pd.DataFrame, columns: list[str], labels: np.ndarray) -> float:
    """
    Función que devuelve al suma de la dispersión de las columnas para cada cluster
    definido en labels

    Parameters
    ----------
    df: pd.DataFrame
        Tabla con las estrellas a clasificar.
    columns: list[str]
        Lista de columnas usadas para medir la desviación típica.
    labels: np.ndarray
        Etiquetas con la clasificación de las estrellas.

    Returns
    -------
    score: float
        Media de la media de la desviación típica normalizada por columna y cluster.
    """
    unique_lab = np.unique(labels[labels > -1])
    std_array = np.zeros((unique_lab.size, len(columns)))

    for i, lab in enumerate(unique_lab):
        gc = df[labels == lab]
        std_array[i, :] = gc[columns].std().values

    score = std_array.sum(axis=1)
    return np.median(score)


def get_params_model_trial(
    trial: Trial, params_distribution: dict[str, list[Union[str, int, float]]]
) -> dict[str, Any]:
    """
    Función que genera los valores del intento trial según la distirbución asociada.
    Parameters
    ----------
    trial: Trial
        Iteración de optuna
    params_distribution:  dict[str, list[Union[str, int, float]]]
        Diccionario con las distribuciones de los parámetros.

    Returns
    -------
    params_model: dict[str, Any]
        Diccionario con los parámetros del modelo.
    """
    params_model = {}
    for param_name, param_distribution in params_distribution.items():
        params_model[param_name] = get_distribution_trial(trial, param_distribution)
    return params_model


def get_distribution_trial(trial: Trial, trial_values: list[Union[str, int, float]]) -> any:
    """
    Método que dado una lista con los valores del tipo de distribución y sus valores
    nos devuelve el objeto instanciado.
    Parameters
    ----------
    trial: Trial
        Trial de optuna.
    trial_values: List[Union[str, int, float]]. Lista de valores donde:
        0. Indica la distribución que siguen los parámetros.
            - "unif": Se utiliza trial.suggest_float.
            - "lunif": Se utiliza trial.suggest_float(log=True).
            - "int": Se utiliza trial.suggest_int.
            - "cat": Se utiliza trial.suggest_categorical.
        *. Resto de valores son los parámetros asociados al
           trial.suggest_ correspondiente a la distribución.
    Returns
    -------
    : Any
        Parámetro del intento asociado.
    """
    distribution = trial_values[0]
    dic_trials_distribution = {
        "unif": trial.suggest_float,
        "lunif": lambda *args: trial.suggest_float(*args, log=True),
        "int": trial.suggest_int,
        "cat": trial.suggest_categorical,
    }
    try:
        return dic_trials_distribution[distribution](*trial_values[1:])
    except KeyError:
        raise DistributionTrialDontExist(
            f"La distribución {distribution} no existe. "
            "Utiliza algunos de los listados: "
            f"{list(dic_trials_distribution.keys())}"
        )


def get_cluster_method(method: ClusterMethodsNames) -> ClusterMethods:
    """
    Función que nos devuelve el modelo de clusterizaación.

    Parameters
    ----------
    method: ClusterMethodsNames
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
    }
    try:
        return methods[method]
    except KeyError:
        raise ValueError(f"No existe el método {method}, prueba con : {list(methods.keys())}")


def get_default_params_distribution(method: ClusterMethodsNames) -> ClusterMethods:
    """
    Función que nos devuelve la distribución por defecto de los parámetros.

    Parameters
    ----------
    method: ClusterMethodsNames
        Nombre del método a utilizar.

    Returns
    -------
    params_distribucion: ClusterParamsDistribution
        Distribución de los parámetros por defecto
    """

    methods = {
        ClusterMethodsNames.DBSCAN_NAME: ClusterParamsDistribution.DBSCAN_PARAMS,
        ClusterMethodsNames.KMEANS_NAME: ClusterParamsDistribution.KMEANS_PARAMS,
        ClusterMethodsNames.HDBSCAN_NAME: ClusterParamsDistribution.HDBSCAN_PARAMS,
        ClusterMethodsNames.GM_NAME: ClusterParamsDistribution.GM_PARAMS,
    }
    try:
        return methods[method]
    except KeyError:
        raise ValueError(f"No existe el método {method}, prueba con : {list(methods.keys())}")


@attrs(auto_attribs=True)
class ClusteringResults:
    """
    Clase que guarda los resultados de la clusterización.
    """

    df_stars: pd.DataFrame
    columns: list[str]
    labels: np.ndarray[int]
    clustering: ClusterMethods

    _storage = ContainerSerializerZip(
        serializers={
            "df_stars": StorageObjectPandasCSV(),
            "columns": StorageObjectPickle(),
            "labels": StorageObjectPickle(),
            "clustering": StorageObjectPickle(),
        }
    )

    def __str__(self) -> str:
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise_ = list(self.labels).count(-1)
        description = f"Estimated number of clusters: {n_clusters_}\n"
        description += f"Estimated number of noise points: {n_noise_}\n"

        for label in np.unique(self.labels[self.labels > -1]):
            mask_i = self.labels == label
            description += f"\t - Volumen total del cluster {label}: {mask_i.sum()}.\n"
        return description

    def save(self, path: str):
        """
        Método que guarda los reusltados en un archivo zip.

        Parameters
        ----------
        path: str
            Ruta donde se quiere guardar los archivos.
        """
        path_name = os.path.join(path, "stars_clustering")
        container = {
            "df_stars": self.df_stars,
            "columns": self.columns,
            "labels": self.labels,
            "clustering": self.clustering,
        }
        self._storage.save(path_name, container)

    @classmethod
    def load(cls, path: str) -> "ClusteringResults":
        """
        Método que carga los resultados de la clusterización.

        Parameters
        ----------
        path: str
            Ruta donde se quiere guardar

        Returns
        -------

        """
        path_name = os.path.join(path, "stars_clustering.zip")
        params = cls._storage.load(path_name)
        return cls(**params)


def optimize_clustering(
    df_stars: pd.DataFrame,
    columns: list[str] = DEFAULT_COLS,
    columns_to_clus: list[str] = DEFAULT_COLS_CLUS,
    max_cluster: int = 10,
    n_trials: int = 100,
    method: ClusterMethodsNames = ClusterMethodsNames.DBSCAN_NAME,
    params_to_opt: Optional[dict[str, list[str, float, int, list[Any]]]] = None,
) -> ClusteringResults:
    """
    Función que clusteriza los datos del catálogo usando las columnas columns_to_clus
    minimizando la desviación típica intercluster de las columns.

    Parameters
    ----------
    df_stars: pd.DataFrame,
        Tabla con las estrellas
    columns: list[str],
        Columnas a calcular la desviación típica.
    max_cluster: int = 10,
        Número máximo de clusters.
    n_trials: int = 100,
        Número de intentos en la optimización.
    columns_to_clus: list[str]
        Columnas usadas en la clusterización.
    method: str, default dbscan
        Método de clusterización utilizado.
    params_to_opt:  dict[str, list[str,float, int, list[Any]]]
        Parámetros a optimizar.

    Returns
    -------
    results: ClusteringResults
        Resultados de la optimización de clusterización
    """

    data = StandardScaler().fit_transform(df_stars[columns_to_clus])
    mask_nan = df_stars[columns_to_clus].isna().any(axis=1).values
    df_stars = df_stars[~mask_nan]
    data = data[~mask_nan]
    clustering_class = get_cluster_method(method)

    if params_to_opt is None:
        params_to_opt = {}

    params_distribution = get_default_params_distribution(method)
    params_distribution.update(params_to_opt)

    def objective(trial: Trial) -> float:
        params = get_params_model_trial(trial, params_distribution)
        clustering = clustering_class(**params)
        clustering.fit(data)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        penalty_clusters = 0 if n_clusters_ < max_cluster else n_clusters_
        if len(set(labels[labels > -1])) > 1:
            return score_cluster(df_stars, columns, labels) + penalty_clusters
        return 1e6

    study = create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logging.info(
        f"Los mejores parámetros encontrados, con un score {study.best_value} "
        f"en la iteración {study.best_trial.number} son:\n"
    )
    for key, param in best_params.items():
        logging.info(f"\t {key}: {param}")

    clustering = clustering_class(**best_params)
    clustering.fit(data)

    results = ClusteringResults(df_stars, columns, clustering.labels_, clustering)
    return results
