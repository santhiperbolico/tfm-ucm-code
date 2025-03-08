import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import StorageObjectFigures
from hyper_velocity_stars_detection.sources.source import AstroObject, AstroObjectData, get_radio
from hyper_velocity_stars_detection.sources.xray_source import XSource
from hyper_velocity_stars_detection.tools.cluster_detection import (
    DEFAULT_COLS,
    DEFAULT_COLS_CLUS,
    ClusteringResults,
    ClusterMethodsNames,
    optimize_clustering,
)
from hyper_velocity_stars_detection.tools.cluster_representations import (
    cluster_representation_with_hvs,
    cmd_with_cluster,
)


@attrs
class AstroObjectProject:
    astro_object = attrib(type=AstroObject, init=True)
    path = attrib(type=str, init=True)
    data_list = attrib(type=list[AstroObjectData], init=True)
    xsource = attrib(type=pd.DataFrame, init=True)
    clustering_results = attrib(type=ClusteringResults, init=True, default=None)

    def __str__(self) -> str:
        """
        Descripción del objeto.
        """
        description = f"Las muestras analizadas de {self.astro_object.name} son:\n"
        for data in self.data_list:
            description += str(data) + "\n"
        description += f"Se han encontrado {self.xsource.shape[0]} fuentes de rayos X.\n"
        if isinstance(self.clustering_results, ClusteringResults):
            description += str(self.clustering_results)
        return description

    @classmethod
    def load_project(cls, name: str, path: str) -> "AstroObjectProject":
        """
        Método que carga los datos de las muestras seleccionadas para un proyecto
        asociado a un objeto astronómico.

        Parameters
        ----------
        path: str
            Directorio donde se encuentran los archivos zip con las muestras.
            El nombre de la carpeta que contiene los zip debe ser el del objeto o proyecto.

        Returns
        -------
        object: AstroObjectProject
            Objeto asociado al proyecto.
        """
        path_project = os.path.join(path, name)
        zip_files = [
            file for file in os.listdir(path_project) if ".zip" == file[-4:] and name in file
        ]
        zip_files.sort()
        data_list = []

        for file in zip_files:
            file_path = os.path.join(path_project, file)
            data_list.append(AstroObjectData.load(file_path))

        astro_object = AstroObject.get_object(name)
        ra = astro_object.coord.ra.value
        dec = astro_object.coord.dec.value
        radius = get_radio(astro_object.info, 1)
        try:
            xsource = XSource.load(path_project)
        except (FileNotFoundError, IsADirectoryError):
            logging.info("No se ha encontrado fuentes de rayos X, se van a descargar.")
            xsource = XSource.download_data(ra, dec, radius)

        try:
            clustering_result = ClusteringResults.load(path_project)
        except (FileNotFoundError, IsADirectoryError):
            clustering_result = None
            logging.info("No se ha encontrado resultados de clustering en el proyecto.")
        return cls(astro_object, path, data_list, xsource, clustering_result)

    def save_project(self, path: Optional[str] = None):
        """
        Método que guarda los resultados de un proyecto en el directorio path dentro de la
        carpeta <name>.

        Parameters
        ----------
        path: Optional[str], default None
            Directorio donde se quiere guardar el proyecto
        """
        if path is None:
            path = self.path

        path_project = os.path.join(path, self.astro_object.name)
        if not os.path.exists(path_project):
            os.mkdir(path_project)

        for data in self.data_list:
            data.save(path=path_project)

        XSource.save(path_project, self.xsource)

        if isinstance(self.clustering_results, ClusteringResults):
            self.clustering_results.save(path_project)

    def get_data(self, data_name: str, index_data: Optional[int] = None) -> pd.DataFrame:
        """
        Métod que dado el nombre y la key del astro data devuelve la tabla.

        Parameters
        ----------
        data_name:str,
            Nombre del astro data que se quiere utilizar.
        index_data: Optional[int] = None,
            Key del astrdata que se quiere utilizar.

        Returns
        -------
        df_r: pd.DataFrame
            Tabla de datos con las estrellas del campo visual a evaluar.
        """
        if isinstance(index_data, int):
            return self.data_list[index_data].get_data(data_name)
        for astro_data in self.data_list:
            if data_name in astro_data.data:
                return astro_data.get_data(data_name)
        raise ValueError(f"El catalogo {data_name} no se ha encontrado en el proyecto")

    def cluster_detection(
        self,
        data_name: str,
        index_data: Optional[int] = None,
        columns: list[str] = DEFAULT_COLS,
        columns_to_clus: list[str] = DEFAULT_COLS_CLUS,
        max_cluster: int = 10,
        n_trials: int = 100,
        method: ClusterMethodsNames = ClusterMethodsNames.DBSCAN_NAME,
        params_to_opt: Optional[dict[str, list[str, float, int, list[Any]]]] = None,
    ) -> ClusteringResults:
        """
        Método que ejecuta la optimización del método de clusterización y guarda el
        elemento de clustering.

        Parameters
        ----------
        data_name:str,
            Nombre del astro data que se quiere utilizar.
        index_data: Optional[int] = None,
            Key del astrdata que se quiere utilizar.
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
        df_stars = self.get_data(data_name, index_data)
        self.clustering_results = optimize_clustering(
            df_stars=df_stars,
            columns=columns,
            columns_to_clus=columns_to_clus,
            max_cluster=max_cluster,
            n_trials=n_trials,
            method=method,
            params_to_opt=params_to_opt,
        )
        return self.clustering_results

    def plot_cluster(
        self,
        hvs_candidates_name: str,
        index_hvs_candidates: Optional[int] = None,
        factor_sigma: float = 1.0,
        hvs_pm: float = 150,
        legend: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Función que representa el cluster con las candidatas HVS en coordenadas galacticas
        y con los vectores de proper motion.

        Parameters
        ----------
        hvs_candidates_name:str,
            Nombre del astro data que se quiere utilizar para seleccionar las HVS.
        index_hvs_candidates: Optional[int] = None,
            Key del astrdata que se quiere utilizar para seleccionar las HVS.
        factor_sigma: float, default 1
            Proporción del sigma del paralaje que se quiere usar para seleccionar las HVS
        hvs_pm: float, default
            Movimiento propio mínimo en km por segundo en la selección de HVS
        legend: bool, default True
            Indica si se quiere graficar la leyenda.
        Returns
        -------
        fig: Figure
            Figura con la representación en coordenadas galactics
        ax: Axes
            Eje de la figura.
        """
        path_project = os.path.join(self.path, self.astro_object.name)
        df_hvs_candidates = self.get_data(hvs_candidates_name, index_hvs_candidates)
        df_gc = self.clustering_results.gc
        df_source_x = self.xsource
        fig, ax = cluster_representation_with_hvs(
            df_gc=df_gc,
            df_hvs_candidates=df_hvs_candidates,
            factor_sigma=factor_sigma,
            hvs_pm=hvs_pm,
            df_source_x=df_source_x,
            legend=legend,
        )
        ax.set_title(f"Cluster {hvs_candidates_name} hvs > {hvs_pm} km/s")
        StorageObjectFigures.save(
            path=os.path.join(path_project, f"cluster_{hvs_candidates_name}_hvs_{hvs_pm}"),
            value=fig,
        )
        return fig, ax

    def plot_cmd(
        self,
        hvs_candidates_name: str,
        index_hvs_candidates: Optional[int] = None,
        factor_sigma: float = 1.0,
        hvs_pm: float = 150,
        df_isochrone: Optional[pd.DataFrame] = None,
        color_field: str = "bp_rp",
        magnitud_field: str = "phot_g_mean_mag",
        isochrone_distance_module: float = 0,
        isochrone_redding: float = 0,
        legend: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Función que genera la gráfica del Color Magnitud Diagram. S

        Parameters
        ----------
        hvs_candidates_name:str,
            Nombre del astro data que se quiere utilizar para seleccionar las HVS.
        index_hvs_candidates: Optional[int] = None,
            Key del astrdata que se quiere utilizar para seleccionar las HVS.
        factor_sigma: float, default 1
            Proporción del sigma del paralaje que se quiere usar para seleccionar las HVS
        hvs_pm: float, default
            Movimiento propio mínimo en km por segundo en la selección de HVS
        df_isochrone: pd.DataFrame
            Tabla con los datos de la isochrona.
        color_field: str
            Nombre del campo de color del CMD.
        magnitud_field: str
            Nombre del campo de la magnitud.
        isochrone_distance_module: float
            Módulo de distancia para la corrección de la isochrona.
        isochrone_redding: float
            Ajuste de enrojecimiento de la isochrona.
        legend: bool, default True
            Indica si se quiere graficar la leyenda.

        Returns
        -------
        fig: Figure
            Figura con el CMD.
        ax: Axes
            Eje de la gráfica.
        """
        path_project = os.path.join(self.path, self.astro_object.name)
        df_hvs_candidates = self.get_data(hvs_candidates_name, index_hvs_candidates)
        selected = self.clustering_results.selected_hvs(df_hvs_candidates, factor_sigma, hvs_pm)

        if isinstance(self.clustering_results, ClusteringResults):
            fig, ax = cmd_with_cluster(
                df_catalog=self.clustering_results.df_stars,
                labels=self.clustering_results.labels,
                df_isochrone=df_isochrone,
                color_field=color_field,
                magnitud_field=magnitud_field,
                isochrone_distance_module=isochrone_distance_module,
                isochrone_redding=isochrone_redding,
            )
            ax.scatter(
                x=selected[color_field],
                y=selected[magnitud_field],
                s=20,
                c="b",
                marker="s",
                label="HVS Candidate",
            )
            if legend:
                plt.legend()
            ax.set_title(f"CMD with {hvs_candidates_name} hvs > {hvs_pm} km/s")
            StorageObjectFigures.save(
                path=os.path.join(path_project, f"cmd_hvs_{hvs_pm}"), value=fig
            )
            return fig, ax
        raise RuntimeError("Genera un clustering antes de ejecutar este método.")
