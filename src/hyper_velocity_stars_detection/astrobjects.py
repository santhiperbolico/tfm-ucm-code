import logging
import os
from typing import Any, Optional

import pandas as pd
from attr import attrib, attrs

from hyper_velocity_stars_detection.sources.source import AstroObject, AstroObjectData, get_radio
from hyper_velocity_stars_detection.sources.xray_source import XSource
from hyper_velocity_stars_detection.tools.cluster_detection import (
    DEFAULT_COLS,
    DEFAULT_COLS_CLUS,
    ClusteringResults,
    ClusterMethodsNames,
    optimize_clustering,
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
