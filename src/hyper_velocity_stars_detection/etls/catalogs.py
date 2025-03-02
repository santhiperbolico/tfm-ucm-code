import logging
import os.path
from typing import Optional

import pandas as pd
from astroquery.esa.xmm_newton import XMMNewton
from astroquery.gaia import Gaia
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import (
    StorageObjectPandasCSV,
    StorageObjectTableVotable,
)


class CatalogsType:
    GAIA_DR2 = "gaiadr2"
    GAIA_DR3 = "gaiadr3"


class XSourceType:
    XMMNEWTON = "XMMNewton"


class CatalogsTables:
    GAIA_DR2 = "gaiadr2.gaia_source"
    GAIA_DR3 = "gaiadr3.gaia_source"


XSOURCES = {XSourceType.XMMNEWTON: XMMNewton}


class CatalogError(Exception):
    pass


@attrs
class Catalog:
    catalog_table = attrib(type=str)

    @classmethod
    def get_catalog(cls, catalog_name: CatalogsType) -> "Catalog":
        """
        Método de clase que dado el nombre del catálogo carga el elemento Catalog
        con su tabla:

        Parameters
        ----------
        catalog_name: CatalogsType
            Nombre del catálogo

        Returns
        -------
        object: Catalog
            Catálogo instanciado.
        """

        dic_types = {
            CatalogsType.GAIA_DR2: CatalogsTables.GAIA_DR2,
            CatalogsType.GAIA_DR3: CatalogsTables.GAIA_DR3,
        }
        try:
            catalog_table = dic_types[catalog_name]
        except KeyError:
            raise CatalogError(f"El catálogo {catalog_name} no existe.")
        return cls(catalog_table)

    def download_results(
        self,
        ra: float,
        dec: float,
        radius: float,
        output_file: str,
        filter_parallax_min: Optional[float] = None,
        filter_parallax_max: Optional[float] = None,
        filter_parallax_error: Optional[float] = None,
        row_limit: int = -1,
    ) -> pd.DataFrame:
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
        output_file: str
            Nombre del archivo a guardar.
        filter_parallax_min: Optional[float], default None
            Paralaje mínimo que buscar.
        filter_parallax_max: Optional[float], default None
            Paralaje máximo que buscar.
        filter_parallax_error: Optional[float], default None
            Máximo error en el parale a buscar.
        row_limit: int, default -1
            Límite de filas consultadas. Por defecto se extraen todas las filas.

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.
        """
        Gaia.ROW_LIMIT = row_limit

        filter = f"""
        WHERE 1=CONTAINS(
               POINT('ICRS', ra, dec),
               CIRCLE('ICRS', {ra}, {dec}, {radius})
               )
        """
        if filter_parallax_max is not None:
            filter = f"{filter} AND parallax < {filter_parallax_max}"
        if filter_parallax_min is not None:
            filter = f"{filter} AND parallax > {filter_parallax_min}"
        if filter_parallax_error is not None:
            filter = f"{filter} AND parallax_error < {filter_parallax_error}"

        query = f"SELECT * FROM {self.catalog_table} {filter}"

        job = Gaia.launch_job_async(
            query, dump_to_file=True, output_format="votable", output_file=f"{output_file}"
        )
        logging.info(job)
        results = job.get_results()
        return results.to_pandas()

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
class XSource:
    results = attrib(type=pd.DataFrame, init=False)

    @staticmethod
    def download_data(ra: float, dec: float, radius: float) -> pd.DataFrame:
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

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.
        """

        query = f"""
        SELECT * FROM v_public_observations
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        )
        """
        # Mostrar los resultados
        discard_cols = [
            "observation_equatorial_spoint",
            "observation_fov_scircle",
            "observation_galactic_spoint",
        ]
        query_results = XMMNewton.query_xsa_tap(query)
        results = query_results[
            [col for col in query_results.columns if col not in discard_cols]
        ].to_pandas()
        return results

    @staticmethod
    def save(path: str, results: pd.DataFrame) -> None:
        path_file = os.path.join(path, "xsource.csv")
        StorageObjectPandasCSV().save(path_file, results)

    @staticmethod
    def load(path: str):
        path_file = os.path.join(path, "xsource.csv")
        results = StorageObjectPandasCSV().load(path_file)
        return results
