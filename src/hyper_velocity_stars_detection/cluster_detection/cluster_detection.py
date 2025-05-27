import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from attr import attrs
from scipy.cluster.hierarchy import fcluster, linkage

from hyper_velocity_stars_detection.cluster_detection.clustering_methods import (
    ClusterMethods,
    get_cluster_method,
)
from hyper_velocity_stars_detection.cluster_detection.preprocessing_methods import (
    NoiseMethod,
    ScalersMethod,
    get_noise_method,
    get_scaler_method,
)
from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageObjectPandasCSV,
    StorageObjectPickle,
)


def get_distance_from_references(
    labels: np.ndarray,
    cluster_data: pd.DataFrame | np.ndarray,
    reference_cluster: pd.Series,
    columns_cluster: list[str] = None,
) -> np.ndarray:
    """
    Función que calcula las distancias para cada uno de los cluster con los datos de referencia.

    Parameters
    ----------
    labels: np.ndarray,
        Etiqueta de cada elemento del cluster data.
    cluster_data: pd.DataFrame | np.ndarray,
        Datos de las estrellas analizados. Si se el pasa un array debe ser con los de las columnas
        correspondientes a columns_cluster.
    reference_cluster: pd.DataFrame
        Datos de referencia del cluster
    columns_cluster: list[str],
        Lista de columnas utilizadas en el clustering

    Returns
    -------
    distances: np.ndarray
        Distancias para cada cluster encontrado.
    """
    if columns_cluster is None:
        columns_cluster = reference_cluster.index
    if isinstance(cluster_data, np.ndarray):
        cluster_data = pd.DataFrame(cluster_data, columns=columns_cluster)
    mean_cluster_dr2 = reference_cluster[columns_cluster].values
    unique_labels = np.unique(labels[labels > -1])
    distances = np.zeros(unique_labels.size)
    for i, label in enumerate(unique_labels):
        gc = cluster_data.loc[labels == label]
        mean_cluster = gc[columns_cluster].mean().values
        distances[i] = np.sqrt(np.linalg.norm(mean_cluster_dr2 - mean_cluster))
    return distances


@attrs(auto_attribs=True)
class ClusteringDetection:
    """
    Clase que engloba el modelo de clusterización, el escalado de los datos y la eliminación
    de outliers.
    """

    model: ClusterMethods
    scaler: ScalersMethod = None
    noise_method: NoiseMethod | None = None
    labels_: np.ndarray = None

    _storage = ContainerSerializerZip(
        serializers={
            "model": StorageObjectPickle(),
            "noise_method": StorageObjectPickle(),
            "scaler": StorageObjectPickle(),
            "labels_": StorageObjectPickle(),
        }
    )

    def fit(
        self,
        x: np.ndarray | pd.DataFrame,
        scaler_params: Optional[dict] = None,
        noise_params: Optional[dict] = None,
        cluster_params: Optional[dict] = None,
    ):
        """
        Método que entrena el scaler, método de eliminación de outliers y clusterización
        asociados.

        Parameters
        ----------
        x: np.ndarray | pd.DataFrame
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
        if isinstance(x_to_fit, pd.DataFrame):
            x_to_fit = x_to_fit.values
        self.labels_ = np.ones(x_to_fit.shape[0]).astype(int)
        if self.scaler is not None:
            x_to_fit = self.scaler.fit_transform(x, **scaler_params)
        if self.noise_method is not None:
            mask = self.noise_method.fit_predict(x_to_fit, **noise_params)
            x_to_fit = x_to_fit[mask > -1]
            self.model.fit(x_to_fit, **cluster_params)
            self.labels_[mask > -1] = self.model.labels_.astype(int)
            self.labels_[mask == -1] = -1
            return None

        self.model.fit(x_to_fit, **cluster_params)
        self.labels_ = self.model.labels_.astype(int)
        return None

    def preprocessing(self, x: np.ndarray | pd.DataFrame):
        """
        Método que preprocesa usando el scaler y método de eliminación de outliers.
        asociados.

        Parameters
        ----------
        x: np.ndarray | pd.DataFrame
            Datos a clusterizar.

        Returns
        -------
        x_pre: np.ndarray
            Preprocesado de los datos.
        """
        x_pre = x.copy()
        if isinstance(x_pre, pd.DataFrame):
            x_pre = x_pre.values
        if self.scaler is not None:
            x_pre = self.scaler.transform(x_pre)
        if self.noise_method is not None:
            mask = self.noise_method.predict(x_pre)
            x_pre = x_pre[mask > -1]
        if isinstance(x, pd.DataFrame):
            x_pre = pd.DataFrame(x_pre, columns=x.columns)
        return x_pre

    def get_main_cluster(
        self,
        cluster_data: Optional[pd.DataFrame] = None,
        reference_cluster: Optional[pd.Series] = None,
        labels: Optional[np.ndarray] = None,
    ) -> int:
        """
        Función que calcula el cluster mayoritario

        Parameters
        ----------
        cluster_data:  Optional[pd.DataFrame] = None
            Datos de las estrellas analizados.
        reference_cluster:  Optional[pd.Series] = None
            Datos de referencia del cluster
        labels: Optional[np.ndarray] = None
            Etiquetas usadas para calcular el cluster principal.

        Returns
        -------
        label_cluster: int
            Valor de la etiqueta del cluster con mayor volumen.
        """
        if not isinstance(labels, np.ndarray):
            labels = self.labels_
        unique_lab = np.unique(labels[labels > -1])
        if unique_lab.size == 0:
            return -1
        j = np.argmax(np.bincount(labels[labels > -1]))
        if isinstance(reference_cluster, pd.Series) and isinstance(cluster_data, pd.DataFrame):
            distances = get_distance_from_references(
                labels=labels,
                cluster_data=self.preprocessing(cluster_data),
                reference_cluster=reference_cluster,
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
        container = {
            "model": self.model,
            "noise_method": self.noise_method,
            "scaler": self.scaler,
            "labels_": self.labels_,
        }
        self._storage.save(path_name, container)

    @classmethod
    def from_cluster_params(
        cls,
        cluster_method: str,
        cluster_params: dict,
        scaler_method: Optional[str] = None,
        noise_method: Optional[str] = None,
    ) -> "ClusteringDetection":
        """
        Método de clase que instacia el Clustering Detection desde los nombres de los tipos
        de clustering, scaler y noise method.

        Parameters
        ----------
        cluster_method: str,
            Nombre del tipo del clustering method.
        cluster_params: dict,
            Parámetros del método de clustering.
        scaler_method: Optional[str] = None,
            Nombre del tipo del scaler.
        noise_method: Optional[str] = None,
            Nombre del tipo del método de eliminación outlier.

        Returns
        -------
        object: ClusteringDetection
            Objeto instanciado.
        """
        clustering = get_cluster_method(cluster_method)(**cluster_params)
        scaler = get_scaler_method(scaler_method)
        noise = get_noise_method(noise_method)
        return cls(clustering, scaler, noise)

    @classmethod
    def load(cls, path: str) -> "ClusteringDetection":
        """
        Método que carga los resultados de la clusterización.

        Parameters
        ----------
        path: str
            Ruta donde se quiere guardar

        Returns
        -------
        object: ClusteringDetection
            Objeto instanciado.
        """
        path_name = os.path.join(path, "clustering_method.zip")
        params = cls._storage.load(path_name)
        return cls(**params)


@attrs(auto_attribs=True)
class ClusteringResults:
    """
    Clase que guarda los resultados de la clusterización.
    """

    df_stars: pd.DataFrame
    columns: list[str]
    columns_to_clus: list[str]
    clustering: ClusteringDetection
    main_label: Optional[int | list[int]] = None

    _storage = ContainerSerializerZip(
        serializers={
            "df_stars": StorageObjectPandasCSV(),
            "columns": StorageObjectPickle(),
            "columns_to_clus": StorageObjectPickle(),
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
        gc = self.df_stars[mask]
        return gc

    @property
    def labels(self) -> np.ndarray:
        """
        Etiquetas de cada una de la muestra de estrellas clusterizada.
        """
        return self.clustering.labels_

    def remove_outliers_gc(self) -> pd.DataFrame:
        """
        Método que devuelve los outliers del cluster principal encontrado.

        Returns
        -------
        gc: pd.DataFrame
            Datos del cluster principal sin outliers.
        """

        gc = self.gc
        if_model = get_noise_method("isolation_forest_method")
        labels = if_model.fit_predict(gc[self.columns_to_clus])
        return gc[labels > -1]

    def set_main_label(
        self,
        main_label: Optional[int | list[int]] = None,
        cluster_data: Optional[pd.DataFrame] = None,
        reference_cluster: Optional[pd.DataFrame] = None,
        group_labels: bool = False,
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
        group_labels: bool, default False
            Indica si se quiere agrupar las etiquetas del cluster
        """
        if main_label is None:
            grouped_labels = None
            if group_labels:
                grouped_labels = self.group_labels()
            main_label = self.clustering.get_main_cluster(
                cluster_data, reference_cluster, grouped_labels
            )
            if group_labels:
                main_label = np.unique(self.labels[grouped_labels == main_label]).tolist()
            self.main_label = main_label
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

    def group_labels(self, threshold: float = 0.5) -> np.ndarray:
        """
        Método que calcula las etiquetas agrupadas.

        Parameters
        ----------
        threshold: float,default 0.5
            Umbral usado en la agrupación de los centroides.

        Returns
        -------
        grouped_lables: np.ndarray
            Etiquetas agrupadas.
        """
        labels = self.get_labels()
        centroids = pd.DataFrame(columns=self.columns)
        for label in labels:
            centroids.loc[label] = self.df_stars.loc[self.labels == label, self.columns].mean()
        centroids_grid = linkage(centroids.values, method="ward")
        cluster_groups = fcluster(centroids_grid, t=threshold, criterion="distance")
        grouped_labels = self.labels.copy()
        for label in cluster_groups:
            grouped_labels[np.isin(grouped_labels, labels[cluster_groups == label])] = label
        return grouped_labels

    def save(self, path: str):
        """
        Método que guarda los reusltados en un archivo zip.

        Parameters
        ----------
        path: str
            Ruta donde se quiere guardar los archivos.
        """
        self.clustering.save(path)
        path_name = os.path.join(path, "stars_clustering")
        container = {
            "df_stars": self.df_stars,
            "columns": self.columns,
            "columns_to_clus": self.columns_to_clus,
            "main_label": self.main_label,
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
        object: ClusteringResults
            Objeto instanciado.
        """
        clustering = ClusteringDetection.load(path)
        path_name = os.path.join(path, "stars_clustering.zip")
        params = cls._storage.load(path_name)
        params["clustering"] = clustering
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
