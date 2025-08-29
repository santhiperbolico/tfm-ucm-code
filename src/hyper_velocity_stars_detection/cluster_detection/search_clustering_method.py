from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from attr import attrs
from optuna.trial import Trial

from hyper_velocity_stars_detection.cluster_detection.cluster_detection import ClusteringDetection
from hyper_velocity_stars_detection.cluster_detection.clustering_methods import (
    get_default_params_distribution,
)


@attrs(auto_attribs=True)
class ParamsDistribution:
    cluster_method: str
    scaler_method_list: list[str | None] | None
    noise_method_list: list[str | None] | None
    params_to_opt: dict[str, list[str | float | int | list[Any]]] | None

    def __str__(self):
        return self.cluster_method

    def get_params_distribution(self, max_cluster: int | None = None) -> dict[str, list[Any]]:
        """
        Método que devuelve la distribución de los parámetros que debe seguir la distribución

        Parameters
        ----------
        max_cluster: int | None
            Si se le indica el número máximo de cluster que debe usar la clusterización.

        Returns
        -------
        params_distribution: dict[str, list[Any]]
            Diccionario con la distribución de los parámetros.
        """
        if self.params_to_opt is None:
            self.params_to_opt = {}
        params_distribution = get_default_params_distribution(
            self.cluster_method, max_cluster, self.params_to_opt
        )
        params_distribution.update(self.params_to_opt)
        if self.scaler_method_list:
            params_distribution["scaler_method"] = [
                "cat",
                "scaler_method",
                self.scaler_method_list,
            ]
        if self.noise_method_list:
            params_distribution["noise_method"] = ["cat", "noise_method", self.noise_method_list]
        return params_distribution


class DistributionTrialDontExist(Exception):
    """
    Error que indica que no existe una distribucion.
    """

    pass


def get_params_model_trial(
    trial: Trial,
    params_distribution: dict[str, list[Union[str, int, float]]],
    method_key: str | None = None,
) -> dict[str, Any]:
    """
    Función que genera los valores del intento trial según la distirbución asociada.
    Parameters
    ----------
    trial: Trial
        Iteración de optuna
    params_distribution:  dict[str, list[Union[str, int, float]]]
        Diccionario con las distribuciones de los parámetros.
    method_key: str | None = None
        Key del método asociado. Se usa para añadir como prefijo del nombre del parámetro.

    Returns
    -------
    params_model: dict[str, Any]
        Diccionario con los parámetros del modelo.
    """
    params_model = {}
    for param_name, param_distribution in params_distribution.items():
        params_model[param_name] = get_distribution_trial(trial, param_distribution, method_key)
    return params_model


def get_distribution_trial(
    trial: Trial, trial_values: list[Union[str, int, float]], method_key: str | None = None
) -> any:
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
    method_key: str | None = None
        Key del método asociado. Se usa para añadir como prefijo del nombre del parámetro.
    Returns
    -------
    : Any
        Parámetro del intento asociado.
    """
    distribution = trial_values[0]
    param_name = trial_values[1]
    if method_key:
        param_name = "%s_%s" % (method_key, param_name)
    dic_trials_distribution = {
        "unif": trial.suggest_float,
        "lunif": lambda *args: trial.suggest_float(*args, log=True),
        "int": trial.suggest_int,
        "cat": trial.suggest_categorical,
    }
    try:
        return dic_trials_distribution[distribution](param_name, *trial_values[2:])
    except KeyError:
        raise DistributionTrialDontExist(
            f"La distribución {distribution} no existe. "
            "Utiliza algunos de los listados: "
            f"{list(dic_trials_distribution.keys())}"
        )


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
class ParamsOptimizator:
    params_methods: list[ParamsDistribution]

    @property
    def list_methods(self) -> list[str]:
        """
        Lista de las keys de los métodos a optimizar.
        """
        return [str(method) for method in self.params_methods]

    def get_params_method(self, method_key: str) -> ParamsDistribution:
        """
        Método que dada una key devuelve los parámetros asociados.

        Parameters
        ----------
        method_key: str
            Key de la distribución de parámetros asociada.

        Returns
        -------
        method: ParamsDistribution
            Distribución de parámetros asociada.
        """
        for method in self.params_methods:
            if str(method) == method_key:
                return method
        raise ValueError("No se ha encontrado %s." % method_key)

    def get_objective_function(
        self,
        df_stars: pd.DataFrame,
        columns: list[str],
        columns_to_clus: list[str],
        max_cluster: int | None = None,
    ) -> Callable[[Trial], float]:
        """
         Función que genera la función objetivo a optimizar.

         Parameters
         ----------
        df_stars: pd.DataFrame,
             Tabla con las estrellas
         columns: list[str],
             Columnas a calcular la desviación típica.
         columns_to_clus: list[str]
             Columnas usadas en la clusterización.
         max_cluster: int | None,
             Número máximo de clusters.

         Returns
         -------
         objective: Callable[[Trial], float]
        """

        def objective(trial: Trial) -> float:
            trial_method_key = trial.suggest_categorical("params_distribution", self.list_methods)
            trial_method = self.get_params_method(trial_method_key)
            params_distribution = trial_method.get_params_distribution(max_cluster)
            params = get_params_model_trial(trial, params_distribution, trial_method_key)
            scaler_method = params.pop("scaler_method", None)
            noise_method = params.pop("noise_method", None)
            clustering_model = ClusteringDetection.from_cluster_params(
                cluster_method=trial_method.cluster_method,
                cluster_params=params,
                scaler_method=scaler_method,
                noise_method=noise_method,
            )
            clustering_model.fit(df_stars[columns_to_clus])
            labels = clustering_model.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            penalty_clusters = 0 if n_clusters_ < max_cluster else n_clusters_
            if len(set(labels[labels > -1])) > 1:
                return score_cluster(df_stars, columns, labels) + penalty_clusters
            return 1e6

        return objective


GM_PARAMS_OPTIMIZATOR = ParamsOptimizator(
    [
        ParamsDistribution(
            "gaussian_mixture",
            ["standard", None],
            [None],
            None,
        )
    ]
)


DEFAULT_PARAMS_OPTIMIZATOR = ParamsOptimizator(
    [
        ParamsDistribution(
            "dbscan",
            ["standard"],
            [None],
            None,
        ),
        ParamsDistribution(
            "hdbscan",
            ["standard"],
            [None],
            None,
        ),
        ParamsDistribution(
            "gaussian_mixture",
            ["standard", "minmax", None],
            ["isolation_forest_method", "local_outlier_method", None],
            None,
        ),
    ]
)
