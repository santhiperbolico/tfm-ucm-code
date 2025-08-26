import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from astroquery.gaia import Gaia

from hyper_velocity_stars_detection.data_storage import StorageObjectTableVotable
from hyper_velocity_stars_detection.sources.filter_quality import (
    GAIA_DR2_FIELDS,
    GAIA_DR3_FIELDS,
    GAIA_FPR_FIELDS,
    QueryProcessor,
)
from hyper_velocity_stars_detection.sources.ruwe_tools.dr2.ruwetools import U0Interpolator


class Catalog(ABC):
    catalog_name = "catalog_name"
    catalog_table = "table_name"

    def download_data(
        self,
        ra: float,
        dec: float,
        radius: float,
        row_limit: int = -1,
        output_file: Optional[str] = None,
        **filter_params,
    ) -> Optional[pd.DataFrame]:
        """
        Método que descarga los datos del catalogo asociado teniendo en cuenta los parámetros.

        Parameters
        ----------
        ra: float
            Ascensión recta en mas.
        dec: float
            Declinaje en mas
        radius: float
            Radio de búsqueda en grados.
        output_file: Optional[str] = None,
            Nombre del archivo donde se quiere guardar los resultados, si se quiere.
        row_limit: int, default -1
            Límite de filas consultadas. Por defecto se extraen todas las filas.
        **filter_params
            Filtros aplicados a la descarga de datos.

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.
        """
        try:
            return self._download_data(
                ra=ra,
                dec=dec,
                radius=radius,
                row_limit=row_limit,
                output_file=output_file,
                **filter_params,
            )
        except NotImplementedError:
            raise NotImplementedError(
                "No se ha implementado la descarga en el catálogo %s" % self.catalog_name
            )

    @abstractmethod
    def _download_data(
        self,
        ra: float,
        dec: float,
        radius: float,
        row_limit: int = -1,
        output_file: Optional[str] = None,
        **filter_params,
    ) -> Optional[pd.DataFrame]:
        """
        Método oculto asociado a download_data.
        """
        raise NotImplementedError()

    @staticmethod
    def read_catalog(file_catalog: str) -> pd.DataFrame:
        """
        Método que lee el archivo asociado al catálogo y lo convierte en DataFrame

        Parameters
        ----------
        file_catalog: str
            Nombre del archivo .vot con los dato sdel catálogo.

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.

        """
        results = StorageObjectTableVotable.load(file_catalog).to_pandas()
        return results


class GaiaDR3(Catalog):
    catalog_name = "gaiadr3"
    catalog_table = "gaiadr3.gaia_source"

    def _download_data(
        self,
        ra: float,
        dec: float,
        radius: float,
        row_limit: int = -1,
        output_file: Optional[str] = None,
        **filter_params,
    ) -> Optional[pd.DataFrame]:
        """
        Método que descarga los datos del catalogo asociado teniendo en cuenta los parámetros.

        Parameters
        ----------
        ra: float
           Ascensión recta en mas.
        dec: float
           Declinaje en mas
        radius: float
           Radio de búsqueda en grados.
        row_limit: int, default -1
           Límite de filas consultadas. Por defecto se extraen todas las filas.
        **filter_params
            Filtros aplicados a la descarga de datos.

        Returns
        -------
        results: pd.DataFrame
           Tabla con los datos del catálogo descargados.
        """
        Gaia.ROW_LIMIT = row_limit

        query_params = {"ra": ra, "dec": dec, "radius": radius}
        query_params.update(filter_params)
        query_processor = QueryProcessor(GAIA_DR3_FIELDS, query_params)
        query = query_processor.get_query(self.catalog_table)
        params_job = {}
        if output_file:
            params_job = {
                "dump_to_file": True,
                "output_format": "votable",
                "output_file": output_file,
            }
        job = Gaia.launch_job_async(query, **params_job)
        logging.info(job)

        results = job.get_results()
        return results.to_pandas()


class GaiaDR2(Catalog):
    catalog_name = "gaiadr2"
    catalog_table = "gaiadr2.gaia_source"

    def _download_data(
        self,
        ra: float,
        dec: float,
        radius: float,
        row_limit: int = -1,
        output_file: Optional[str] = None,
        **filter_params,
    ) -> Optional[pd.DataFrame]:
        """
        Método que descarga los datos del catalogo asociado teniendo en cuenta los parámetros.

        Parameters
        ----------
        ra: float
           Ascensión recta en mas.
        dec: float
           Declinaje en mas
        radius: float
           Radio de búsqueda en grados.
        row_limit: int, default -1
           Límite de filas consultadas. Por defecto se extraen todas las filas.
        **filter_params
            Filtros aplicados a la descarga de datos.

        Returns
        -------
        results: pd.DataFrame
           Tabla con los datos del catálogo descargados.
        """
        Gaia.ROW_LIMIT = row_limit

        query_params = {"ra": ra, "dec": dec, "radius": radius}
        query_params.update(filter_params)
        query_processor = QueryProcessor(GAIA_DR2_FIELDS, query_params)
        query = query_processor.get_query(self.catalog_table)

        params_job = {}
        if output_file:
            params_job = {
                "dump_to_file": True,
                "output_format": "votable",
                "output_file": output_file,
            }
        job = Gaia.launch_job_async(query, **params_job)
        logging.info(job)
        results = job.get_results()
        df_data = results.to_pandas()

        ruwe = query_processor.get_field_value("ruwe")
        if ruwe:
            ruwe_field = query_processor.get_field("ruwe")
            u0_object = U0Interpolator()
            ruwe_values = u0_object.get_ruwe_from_gaia(df_data)
            if ruwe_field.operation == "ls":
                df_data = df_data[ruwe_values < ruwe]
            if ruwe_field.operation == "gs":
                df_data = df_data[ruwe_values > ruwe]

        return df_data


class GaiaFPR(Catalog):
    catalog_name = "gaiafpr"
    catalog_table = "gaiafpr.crowded_field_source"

    def _download_data(
        self,
        ra: float,
        dec: float,
        radius: float,
        row_limit: int = -1,
        output_file: Optional[str] = None,
        **filter_params,
    ) -> Optional[pd.DataFrame]:
        """
        Método que descarga los datos del catalogo asociado teniendo en cuenta los parámetros.

        Parameters
        ----------
        ra: float
           Ascensión recta en mas.
        dec: float
           Declinaje en mas
        radius: float
           Radio de búsqueda en grados.
        row_limit: int, default -1
           Límite de filas consultadas. Por defecto se extraen todas las filas.
        **filter_params
            Filtros aplicados a la descarga de datos.

        Returns
        -------
        results: pd.DataFrame
           Tabla con los datos del catálogo descargados.
        """
        Gaia.ROW_LIMIT = row_limit

        query_params = {"ra": ra, "dec": dec, "radius": radius}
        query_params.update(filter_params)
        query_processor = QueryProcessor(GAIA_FPR_FIELDS, query_params)
        query = query_processor.get_query(self.catalog_table)

        params_job = {}
        if output_file:
            params_job = {
                "dump_to_file": True,
                "output_format": "votable",
                "output_file": output_file,
            }
        job = Gaia.launch_job_async(query, **params_job)
        logging.info(job)
        results = job.get_results()
        df_data = results.to_pandas()

        return df_data
