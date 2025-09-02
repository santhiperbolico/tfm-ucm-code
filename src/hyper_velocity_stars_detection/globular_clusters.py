import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from attr import attrib, attrs
from optuna import create_study

from hyper_velocity_stars_detection.cluster_detection.cluster_detection import (
    CLUSTERING_RESULTS,
    ClusteringDetection,
    ClusteringResults,
)
from hyper_velocity_stars_detection.cluster_detection.search_clustering_method import (
    DEFAULT_PARAMS_OPTIMIZATOR,
    ParamsOptimizator,
)
from hyper_velocity_stars_detection.data_storage import (
    ContainerSerializerZip,
    StorageCustomObject,
    StorageObjectFigures,
)
from hyper_velocity_stars_detection.default_variables import (
    DEFAULT_COLS,
    DEFAULT_COLS_CLUS,
    DEFAULT_ITERATIONS,
    ERROR_COLUMNS,
    MAX_CLUSTER_DEFAULT,
    MAX_SAMPLE_OPTIMIZE,
)
from hyper_velocity_stars_detection.sources.source import AstroMetricData
from hyper_velocity_stars_detection.sources.xray_source import XRSourceData
from hyper_velocity_stars_detection.tools.cluster_representations import (
    cluster_representation,
    cmd_plot,
    cmd_with_cluster,
)
from hyper_velocity_stars_detection.variables_names import (
    ASTRO_OBJECT_DATA,
    BP_RP,
    G_MAG,
    GLOBULAR_CLUSTER_ANALYSIS,
    XRSOURCE,
)


def cluster_detection(
    df_data: pd.DataFrame,
    cluster_method: str,
    cluster_params: Optional[dict[str, Any]] = None,
    scaler_method: str | None = None,
    noise_method: str | None = None,
    columns: Optional[list[str]] = None,
    columns_to_clus: Optional[list[str]] = None,
    reference_cluster: Optional[pd.Series] = None,
    group_labels: bool = False,
) -> ClusteringResults:
    """
    Método que ejecuta la optimización del método de clusterización y guarda el
    elemento de clustering.

    Parameters
    ----------
    df_data: pd.DataFrame,
        Datos utilizados en la detección del cluster
    columns: list[str],
        Columnas a calcular la desviación típica.
    columns_to_clus: list[str]
        Columnas usadas en la clusterización.
    cluster_method: str, default dbscan
        Método de clusterización utilizado.
    scaler_method: str | None = None,
        Método utilizado para escalar los datos.
    noise_method: str | None = None,
        Método utilizado para eliminar ruido de la muestra.
    cluster_params: Optional[dict[str, list[str | float | int | list[Any]]]]
        Parámetros del método de clusterización.
    reference_cluster: Optional[pd.Series] = None
        Datos de refrencia para buscar el cluster que más se aproxime a estos datos.
    group_labels: bool = False,
        Indica si se quiere agrupar las etiquetas parecidas.

    Returns
    -------
    results: ClusteringResults
        Resultados de la optimización de clusterización
    """
    if cluster_params is None:
        cluster_params = {}
    if columns is None:
        columns = DEFAULT_COLS
    if columns_to_clus is None:
        columns_to_clus = DEFAULT_COLS_CLUS

    clustering_best = ClusteringDetection.from_cluster_params(
        cluster_method=cluster_method,
        cluster_params=cluster_params,
        scaler_method=scaler_method,
        noise_method=noise_method,
    )
    clustering_best.fit(df_data[columns_to_clus])
    clustering_results = ClusteringResults(
        df_stars=df_data,
        columns=columns,
        columns_to_clus=columns_to_clus,
        clustering=clustering_best,
        main_label=None,
    )
    clustering_results.set_main_label(
        main_label=None,
        cluster_data=df_data[columns_to_clus],
        reference_cluster=reference_cluster,
        group_labels=group_labels,
    )
    return clustering_results


@attrs
class GlobularClusterAnalysis:
    """
    Clase que recoge los datos de un cúmulo globular y es capaz de implementar métodos
    para la detección de sus estrellas y su representación.
    """

    astro_data = attrib(type=AstroMetricData, init=True)
    xrsource = attrib(type=XRSourceData, init=True)
    clustering_results = attrib(type=Optional[ClusteringResults], init=True, default=None)

    _storage = ContainerSerializerZip(
        serializers={
            ASTRO_OBJECT_DATA: StorageCustomObject(AstroMetricData, ASTRO_OBJECT_DATA),
            XRSOURCE: StorageCustomObject(XRSourceData, XRSOURCE),
            CLUSTERING_RESULTS: StorageCustomObject(ClusteringResults, CLUSTERING_RESULTS),
        }
    )

    def __str__(self) -> str:
        """
        Descripción del objeto.
        """
        description = str(self.astro_data)
        description += f"Se han encontrado {self.xrsource.data.shape[0]} fuentes de rayos X.\n"
        if isinstance(self.clustering_results, ClusteringResults):
            description += str(self.clustering_results)
        return description

    @property
    def name(self) -> str:
        return self.astro_data.astro_object.main_id

    @property
    def df_stars(self) -> pd.DataFrame:
        if isinstance(self.clustering_results, ClusteringResults):
            return self.clustering_results.gc
        raise ValueError(
            "No se han detectado estrellas del cluster, asegurate de haber "
            "ejecutado el clustering."
        )

    @classmethod
    def load_globular_cluster(
        cls,
        name: str,
        catalog_name: str,
        xrs_catalog_name: Optional[str | list[str]] = None,
        radius: Optional[float] = None,
        radius_scale: float = 1.0,
        clustering_detection: bool = False,
        **filter_params,
    ) -> "GlobularClusterAnalysis":
        """
        Método de clase que carga un cúmulo globular descargando lso datos astrométricos y de
        rayos encontrados. Si se le indica, implementa la detección de las estrellas del cúmulo
        a traves de algoritmos de clusterización.

        Parameters
        ----------
        name: str
            Nombre del objeto.
        catalog_name: str
            Nombre del catálogo utilizado para descargarlso datos
        xrs_catalog_name: Optional[str | list[str]]= None
            Nombre del catálogo de Rayos X utilizado para descargarlso datos
        radius: Optional[float] = None
            Radio de búsqueda en grados. SI no se indica se utilizará radius_scale.
        radius_scale: float = 1.0
            Escala del radio de búsqueda en comparación con el campo de visión del objeto.
            El campo de visión es extraido de la base de datos de Simbad.
            Solo se utiliza si no se indica el radius
        clustering_detection: bool = False,
            Indica si se quiere aplicar la detección de las estrellas del cluster con
            los parámetros por defecto.
        **filter_params:
            Filtros aplicados a la descarga de datos astrométricos.

        Returns
        -------
        object_data: GlobularClusterAnalysis
            Clase que recoge las cuatro muestras de la consulta.
        """
        logging.info("-- Descargando datos astrométricos de %s" % name)
        astrometric_data = AstroMetricData.load_data(
            name=name,
            catalog_name=catalog_name,
            radius=radius,
            radius_scale=radius_scale,
            **filter_params,
        )
        logging.info("-- Descargando datos de fuentes de rayos X de %s" % name)
        xrsource = XRSourceData.load_data(
            name=name, catalog_name=xrs_catalog_name, radius=radius, radius_scale=radius_scale
        )
        gc_object = cls(astrometric_data, xrsource)
        if clustering_detection:
            logging.info("-- Detectando las estrellas pertenecientes al cúmulo")
            gc_object.cluster_star_detection()
        return gc_object

    @classmethod
    def load(cls, path: str) -> "GlobularClusterAnalysis":
        """
        Método que carga los datos de las muestras seleccionadas para un proyecto
        asociado a un objeto astronómico.

        Parameters
        ----------
        name: str
            Nombre del objeto de estudio.
        path: str
            Directorio donde se encuentran los archivos zip con las muestras.
            El nombre de la carpeta que contiene los zip debe ser el del objeto o proyecto.
        from_zip: bool = False
            Indica si se quiere cargar desde zip.

        Returns
        -------
        object: GlobularClusterAnalysis
            Objeto asociado al proyecto.
        """
        filepath = path
        if not path.endswith(".zip"):
            filepath = filepath + ".zip"
        container = cls._storage.load(filepath, ignore_errors=True)
        return cls(**container)

    def save(self, path: str):
        """
        Método que guarda los resultados de un proyecto en el directorio path dentro de la
        carpeta <name>.

        Parameters
        ----------
        path: str
            Directorio donde se quiere guardar el proyecto
        """

        filename = f"{GLOBULAR_CLUSTER_ANALYSIS}_{self.name}"
        path_project = os.path.join(path, filename)
        container = {
            ASTRO_OBJECT_DATA: self.astro_data,
            XRSOURCE: self.xrsource,
            CLUSTERING_RESULTS: self.clustering_results,
        }
        self._storage.save(path_project, container, ignore_errors=True)

    def cluster_star_detection(
        self,
        target_columns: Optional[list[str]] = None,
        variables_columns: Optional[list[str]] = None,
        max_cluster: int = MAX_CLUSTER_DEFAULT,
        n_trials: int = DEFAULT_ITERATIONS,
        sample_label: Optional[str] = None,
        max_stars_to_clus: int = MAX_SAMPLE_OPTIMIZE,
        reference_cluster: Optional[pd.Series] = None,
        params_methods: ParamsOptimizator = DEFAULT_PARAMS_OPTIMIZATOR,
        group_labels: bool = False,
    ) -> ClusteringResults:
        """
        Método que ejecuta la optimización del método de clusterización y guarda el
        elemento de clustering.

        Parameters
        ----------
        target_columns: Optional[list[str]] = None,
            Columnas a calcular la desviación típica.
        variables_columns: Optional[list[str]] = None,
            Columnas usadas en la clusterización.
        max_cluster: int = 10,
            Número máximo de clusters.
        n_trials: int = 100,
            Número de intentos en la optimización.
        sample_label: Optional[str] = None,
            Nombre del astro data que se quiere utilizar.
        max_stars_to_clus: int = MAX_SAMPLE_OPTIMIZE,
            Si sample_label es None selecciona la muestra con mayor muestra que no sobrepase
            max_stars_to_clus.
        reference_cluster: Optional[pd.Series] = None
            Datos de refrencia para buscar el cluster que más se aproxime a estos datos.
        params_methods: ParamsOptimizator = DEFAULT_PARAMS_OPTIMIZATOR,
            Lista de la distribución de parámetros que se quiere utilizar.
        group_labels: bool = False,
            Indica si se quiere agrupar las etiquetas parecidas.

        Returns
        -------
        results: ClusteringResults
            Resultados de la optimización de clusterización
        """
        if target_columns is None:
            target_columns = DEFAULT_COLS
        if variables_columns is None:
            variables_columns = DEFAULT_COLS_CLUS

        if sample_label is None:
            sample_label = self.astro_data.get_data_max_samples(max_stars_to_clus)

        df_data = self.astro_data.get_data(sample_label)
        mask_nan = df_data[variables_columns].isna().any(axis=1).values
        df_stars = df_data.loc[~mask_nan, :]

        df_stars_to_clus = df_stars.copy()
        if df_stars_to_clus.shape[0] > max_stars_to_clus:
            df_stars_to_clus = df_stars_to_clus.sort_values(by=ERROR_COLUMNS, ascending=True).iloc[
                :max_stars_to_clus
            ]

        objective = params_methods.get_objective_function(
            df_stars=df_stars_to_clus,
            columns=target_columns,
            columns_to_clus=variables_columns,
            max_cluster=max_cluster,
        )

        study = create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_method_key = best_params.pop("params_distribution")
        best_method = params_methods.get_params_method(best_method_key)

        logging.info(
            f"Los mejores parámetros encontrados, con un score {study.best_value} "
            f"en la iteración {study.best_trial.number} son:\n"
        )
        cluster_params = {}
        for key, param in best_params.items():
            if key.startswith(best_method_key):
                logging.info(f"\t {key}: {param}")
                cluster_params[key.replace("%s_" % best_method_key, "")] = param

        scaler_method = cluster_params.pop("scaler_method", None)
        noise_method = cluster_params.pop("noise_method", None)
        self.clustering_results = cluster_detection(
            df_data=df_stars_to_clus,
            cluster_method=best_method.cluster_method,
            cluster_params=cluster_params,
            scaler_method=scaler_method,
            noise_method=noise_method,
            columns=target_columns,
            columns_to_clus=variables_columns,
            reference_cluster=reference_cluster,
            group_labels=group_labels,
        )
        return self.clustering_results

    def get_cluster_stars(
        self, remove_noise: bool = False, random_state: int | None = None
    ) -> pd.DataFrame:
        """
        Función que devuelve lso elementos detectados pertenecientes al cluster. Si se indica,
        implementa una eliminación de outliers.

        Parameters
        ----------
        remove_noise: bool = False
            Indica si queremos eliminar ruido del cluster principal.
        random_state: int | None = None,
            Semilla en el método de eliminación de ruido del cluster principal.
        Returns
        -------
        df_stars: pd.DataFrame
            Tabla con las estrellas seleccionadas del cluster
        """
        df_gc = self.clustering_results.gc
        if remove_noise:
            df_gc = self.clustering_results.remove_outliers_gc(random_state=random_state)
        return df_gc

    def plot_cluster(
        self,
        sample_label: Optional[str] = None,
        highlight_stars: Optional[pd.DataFrame] = None,
        remove_noise: bool = False,
        random_state: int | None = None,
        path: Optional[str] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Función que representa el cluster con las candidatas HVS en coordenadas galacticas
        y con los vectores de proper motion.

        Parameters
        ----------
        sample_label: Optional[str] = None.
            Si se indica el nombre de la muestra a representar solo se pintará esa muestra,
            ignorando el clustering realizado.
        highlight_stars: Optional[pd.DataFrame] = None
            Tabla con las estrellas que se quieren destacar junto con el cluster
        remove_noise: bool = False
            Indica si queremos eliminar ruido del cluster principal.
        random_state: int | None = None,
            Semilla en el método de eliminación de ruido del cluster principal.
        path: Optional[str] =  None,
            Si se le indica guarda la gráfica en un archivo dentro de path.
        **kwargs:
            Parámetros de la función cluster_representation.
        Returns
        -------
        fig: Figure
            Figura con la representación en coordenadas galactics
        ax: Axes
            Eje de la figura.
        """
        data_name = sample_label
        if not data_name:
            data_name = "df_c1"

        df_gc = self.astro_data.get_data(data_name)
        if isinstance(self.clustering_results, ClusteringResults) and not sample_label:
            df_gc = self.clustering_results.gc
            if remove_noise:
                df_gc = self.clustering_results.remove_outliers_gc(random_state=random_state)
        df_source_x = self.xrsource.data
        df_source_x = df_source_x[df_source_x.main_id == self.name]
        fig, ax = cluster_representation(
            df_gc=df_gc, df_highlights_stars=highlight_stars, df_source_x=df_source_x, **kwargs
        )
        ax.set_title(f"Cluster {self.name}")

        if fig is not None and isinstance(path, str):
            filename = f"cluster_{self.astro_data.data_name}"
            StorageObjectFigures.save(
                path=os.path.join(path, filename),
                value=fig,
            )
        return fig, ax

    def plot_cmd(
        self,
        sample_label: Optional[str] = None,
        highlight_stars: Optional[pd.DataFrame] = None,
        path: Optional[str] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Función que genera la gráfica del Color Magnitud Diagram. S

        Parameters
        ----------
        sample_label: Optional[str] = None.
            Si se indica el nombre de la muestra a representar solo se pintará esa muestra,
            ignorando el clustering realizado.
        highlight_stars: Optional[pd.DataFrame] = None
            Tabla con las estrellas que se quieren destacar junto con el cluster
        path: Optional[str] =  None,
            Si se le indica guarda la gráfica en un archivo dentro de path.
        **kwargs:
            Parámetros de la función cmd_with_cluster.

        Returns
        -------
        fig: Figure
            Figura con el CMD.
        ax: Axes
            Eje de la gráfica.
        """
        data_name = sample_label
        if not data_name:
            data_name = "df_c1"

        switcher_method = {"cmd": cmd_plot, "cmd_cluster": cmd_with_cluster}
        main_method = "cmd"
        message = (
            "-- Genera un clustering si quieres distinguis las estrellas del objeto o no "
            "selecciones muestra."
        )
        params = {"df_catalog": self.astro_data.get_data(data_name)}

        if isinstance(self.clustering_results, ClusteringResults) and not sample_label:
            main_method = "cmd_cluster"
            params = {
                "df_catalog": self.clustering_results.df_stars,
                "labels": self.clustering_results.labels,
            }
            message = "-- Generando el CMD con las estrellas seleccionadas del cluster."

        logging.info(message)
        fig, ax = switcher_method[main_method](**params, **kwargs)

        if isinstance(highlight_stars, pd.DataFrame):
            color_field = kwargs.get("color_field", BP_RP)
            mag_field = kwargs.get("magnitud_field", G_MAG)
            legend = kwargs.get("legend", False)
            ax.scatter(
                x=highlight_stars[color_field],
                y=highlight_stars[mag_field],
                s=20,
                c="b",
                marker="s",
                label="Highlights Stars",
            )
            if legend:
                plt.legend()
        ax.set_title(f"CMD {self.name}")
        if fig is not None and path is not None:
            filename = f"cmd_{self.astro_data.data_name}"
            StorageObjectFigures.save(path=os.path.join(path, filename), value=fig)
        return fig, ax

    def describe(
        self,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if columns is None:
            columns = DEFAULT_COLS_CLUS
        return self.clustering_results.gc[columns].describe()
