import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from astroquery.gaia import Gaia
from attr import attrs

from hyper_velocity_stars_detection.data_storage import StorageObjectTableVotable
from hyper_velocity_stars_detection.sources.filter_quality import GAIA_DR3_FIELDS, QueryProcessor


class CatalogsType:
    GAIA_DR2 = "gaiadr2"
    GAIA_DR3 = "gaiadr3"
    GAIA_FPR = "gaiafpr"


class CatalogsTables:
    GAIA_DR2 = "gaiadr2.gaia_source"
    GAIA_DR3 = "gaiadr3.gaia_source"
    GAIA_FPR = "gaiafpr.crowded_field_source"


class CatalogError(Exception):
    pass


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
            self._download_data(
                ra=ra, dec=dec, radius=radius, row_limit=row_limit, **filter_params
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


@attrs
class GaiaDR3(Catalog):
    catalog_name = "gaiadr3"
    catalog_table = "gaiadr3.gaia_source"

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
        query_processor = QueryProcessor(query_params, GAIA_DR3_FIELDS)
        query = query_processor.get_query(self.catalog_table)

        job = Gaia.launch_job_async(
            query, dump_to_file=True, output_format="votable", output_file=output_file
        )
        logging.info(job)

        results = job.get_results()
        return results.to_pandas()
