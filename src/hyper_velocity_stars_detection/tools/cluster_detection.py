import logging
import os
from typing import Any, Optional, Self, Tuple, Type, Union

import numpy as np
import pandas as pd
from attr import attrs
from optuna import create_study
from optuna.trial import Trial
from sklearn.preprocessing import StandardScaler

from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageObjectPandasCSV,
    StorageObjectPickle,
)
from hyper_velocity_stars_detection.tools.clustering_methods import (
    DBSCAN,
    HDBSCAN,
    ClusterMethodsNames,
    GaussianMixtureClustering,
    KMeans,
    get_cluster_method,
    get_default_params_distribution,
)

ClusterMethods = Type[DBSCAN | KMeans | HDBSCAN | GaussianMixtureClustering]

DEFAULT_COLS = ["pm", "parallax"]
DEFAULT_COLS_CLUS = DEFAULT_COLS + ["bp_rp", "phot_g_mean_mag"]
COLUMNS_CLUSTER = "parallax", "pmra", "pmdec"


class DistributionTrialDontExist(Exception):
    """
    Error que indica que no existe una distribucion.
    """

    pass


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


def get_distance_from_references(
    labels: np.ndarray, cluster_data: pd.DataFrame, reference_cluster: pd.Series
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
    mean_cluster_dr2 = reference_cluster[COLUMNS_CLUSTER].iloc[0, :].values
    unique_labels = np.unique(labels)
    distances = np.zeros(unique_labels.size)
    for i, label in enumerate(unique_labels):
        gc = cluster_data.loc[labels == label]
        mean_cluster = gc[COLUMNS_CLUSTER].mean().values
        distances[i] = np.sqrt(np.linalg.norm(mean_cluster_dr2 - mean_cluster))
    return distances


def get_main_cluster(
    labels: np.ndarray[int],
    cluster_data: Optional[pd.DataFrame] = None,
    reference_cluster: Optional[pd.Series] = None,
) -> int:
    """
    Función que calcula el cluster mayoritario

    Parameters
    ----------
    labels: np.ndarray
        Arrays con las etiquetas asociadas a los cluster
    cluster_data:  Optional[pd.DataFrame] = None
        Datos de las estrellas analizados.
    reference_cluster:  Optional[pd.Series] = None
        Datos de referencia del cluster

    Returns
    -------
    label_cluster: int
        Valor de la etiqueta del cluster con mayor volumen.
    """
    unique_lab = np.unique(labels[labels > -1])
    if unique_lab.size == 0:
        return -1
    j = np.argmax(np.bincount(labels[labels > -1]))
    if isinstance(reference_cluster, pd.Series) and isinstance(cluster_data, pd.DataFrame):
        distances = get_distance_from_references(
            labels[labels > -1], cluster_data, reference_cluster
        )
        j = np.argmin(distances)
    return int(unique_lab[j])


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


@attrs(auto_attribs=True)
class ClusteringResults:
    """
    Clase que guarda los resultados de la clusterización.
    """

    df_stars: pd.DataFrame
    columns: list[str]
    labels: np.ndarray[int]
    clustering: ClusterMethods
    main_label: Optional[int | list[int]] = None

    _storage = ContainerSerializerZip(
        serializers={
            "df_stars": StorageObjectPandasCSV(),
            "columns": StorageObjectPickle(),
            "labels": StorageObjectPickle(),
            "clustering": StorageObjectPickle(),
            "main_label": StorageObjectPickle(),
        }
    )

    def __attrs_post_init__(self):
        if self.main_label is None:
            self.set_main_label()

    def __str__(self) -> str:
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise_ = list(self.labels).count(-1)
        description = f"Estimated number of clusters: {n_clusters_}\n"
        description += f"Estimated number of noise points: {n_noise_}\n"

        for label in np.unique(self.labels[self.labels > -1]):
            mask_i = np.isin(self.labels, label)
            n_cluster = mask_i.sum()
            description += f"\t - Volumen total del cluster {label}: {n_cluster}.\n"
        return description

    @property
    def gc(self) -> pd.DataFrame:
        """
        Estrellas del conjutno de estrellas mayoritario.
        """
        mask = np.isin(self.labels, self.main_label)
        return self.df_stars[mask]

    def set_main_label(
        self,
        main_label: Optional[int | list[int]] = None,
        cluster_data: Optional[pd.DataFrame] = None,
        reference_cluster: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Método que modifica la etiqueta principal del cluster
        Parameters
        ----------
        main_label: Optional[int | list[int]], default None
            Etiqueta que se quiere utilizar ocmo principal. Por defecto se
            utiliza la mayoritaria.
        cluster_data:  Optional[pd.DataFrame] = None
            Datos de las estrellas analizados.
        reference_cluster:  Optional[pd.Series] = None
            Datos de referencia del cluster
        """
        if main_label is None:
            self.main_label = get_main_cluster(self.labels, cluster_data, reference_cluster)
            return None
        self.main_label = main_label
        return None

    def get_labels(
        self, return_counts: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Método que modifica la etiqueta principal del cluster
        Parameters
        ----------
        return_counts: bool = False
            Indica si se quiere devolver el volumen de cada cluster

        Returns
        -------
        unique_labels: np.ndarray
            Etiquetas de los cluster
        counts_labels: np.ndarray, optional
            Si return_counts es True devuelve el volumen de cada cluster en este array.
        """

        return np.unique(self.labels[self.labels > -1], return_counts=return_counts)

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
            "main_label": self.main_label,
        }
        self._storage.save(path_name, container)

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
        path_name = os.path.join(path, "stars_clustering.zip")
        try:
            params = cls._storage.load(path_name)
        except FileNotFoundError as error:
            if "main_label" not in str(error):
                raise error
            serializers = cls._storage._serializers.copy()
            _ = serializers.pop("main_label")
            storage = ContainerSerializerZip(serializers=serializers)
            params = storage.load(path_name)
        return cls(**params)

    def selected_hvs(
        self,
        df_hvs_candidates: pd.DataFrame,
        factor_sigma: float = 1.0,
        hvs_pm: float = 150,
    ):
        """
        Función que filtra y selecciona las candidatas a HVS.

        Parameters
        ----------
        df_hvs_candidates: pd.DataFrame
           Catálogo de estrellas donde se quiere buscar las HVS
        factor_sigma: float, default 1
           Proporción del sigma del paralaje que se quiere usar para seleccionar las HVS
        hvs_pm: float, default
           Movimiento propio mínimo en km por segundo en la selección de HVS

        Returns
        -------
        selected: pd.DataFrame
            Estrellas seleccionadas
        """
        parallax_range = [
            self.gc.parallax.mean() - factor_sigma * self.gc.parallax.std(),
            self.gc.parallax.mean() + factor_sigma * self.gc.parallax.std(),
        ]

        mask_p = (df_hvs_candidates.parallax > parallax_range[0]) & (
            df_hvs_candidates.parallax < parallax_range[1]
        )

        pm_candidates = df_hvs_candidates.pm_kms - self.gc.pm_kms.mean()
        mask_hvs = (pm_candidates > hvs_pm) & mask_p
        return df_hvs_candidates[mask_hvs]


def optimize_clustering(
    df_stars: pd.DataFrame,
    columns: Optional[list[str]] = None,
    columns_to_clus: Optional[list[str]] = None,
    max_cluster: int = 10,
    n_trials: int = 100,
    method: str = ClusterMethodsNames.DBSCAN_NAME,
    params_to_opt: Optional[dict[str, list[str | float | int | list[Any]]]] = None,
    reference_cluster: Optional[pd.Series] = None,
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
    params_to_opt:  dict[str, list[str|float|int|list[Any]]]]
        Parámetros a optimizar.

    Returns
    -------
    results: ClusteringResults
        Resultados de la optimización de clusterización
    """
    if columns is None:
        columns = DEFAULT_COLS
    if columns_to_clus is None:
        columns_to_clus = DEFAULT_COLS_CLUS

    data = StandardScaler().fit_transform(df_stars[columns_to_clus])
    mask_nan = df_stars[columns_to_clus].isna().any(axis=1).values
    df_stars = df_stars[~mask_nan]
    data = data[~mask_nan]
    clustering_class = get_cluster_method(method)

    if params_to_opt is None:
        params_to_opt = {}

    params_distribution = get_default_params_distribution(method, max_cluster, params_to_opt)
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
    main_label = get_main_cluster(clustering.labels_, data, reference_cluster)
    results = ClusteringResults(df_stars, columns, clustering.labels_, clustering, main_label)
    return results
