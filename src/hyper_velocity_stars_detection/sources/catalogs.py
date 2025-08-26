import logging
from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from zero_point import zpt

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

    def fix_parallax_zero_point(self, df_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Método que corrige el sesgo del paralaje de los datos del objeto calculando el punto zero
        del paralaje corregido

        Parameters
        ----------
        df_data: pd.DataFrame
            Datos donde queremos calcular el paralaje corregido.

        Returns
        -------
        parallax_corrected: pd.Series
            Serie de datos con el paralaje corregido.

        """
        try:
            return self._fix_parallax_zero_point(df_data=df_data, **kwargs)
        except NotImplementedError:
            raise NotImplementedError(
                "No se ha implementado la corrección del "
                "paralaje en el catálogo %s" % self.catalog_name
            )

    def _fix_parallax_zero_point(self, df_data: pd.DataFrame, **kwargs) -> pd.Series:
        raise NotImplementedError

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

    def _fix_parallax_zero_point(self, df_data: pd.DataFrame, warnings: bool = True) -> np.ndarray:
        """
        Método que corrige el sesgo del paralaje de los datos del objeto calculando
        el punto zero del paralaje corregido
        (Lindegren et al., 2021 https://doi.org/10.1051/0004-6361/202039653).

        Parameters
        ----------
        df_data: pd.DataFrame
            Datos donde queremos calcular el paralaje corregido.
        warnings: bool, default True
            Indica si se quiere mostrar los warning asociadso a zero_point.get_zpt

        Returns
        -------
        parallax_corrected: np.ndarray
            Array de datos con el paralaje corregido.

        """

        needed_columns = [
            "parallax",
            "phot_g_mean_mag",
            "nu_eff_used_in_astrometry",
            "nu_eff_used_in_astrometry",
            "pseudocolour",
            "ecl_lat",
            "astrometric_params_solved",
        ]
        if not np.isin(needed_columns, df_data.columns).all():
            raise ValueError("Faltan columnas para implementar la corrección del paralaje.")

        zpt.load_tables()
        parallax = df_data["parallax"].values
        phot_g_mean_mag = df_data["phot_g_mean_mag"].values
        nueffused = df_data["nu_eff_used_in_astrometry"].values
        pseudocolour = df_data["pseudocolour"].values
        ecl_lat = df_data["ecl_lat"].values
        astrometric_params_solved = df_data["astrometric_params_solved"].values
        zpvals = zpt.get_zpt(
            phot_g_mean_mag,
            nueffused,
            pseudocolour,
            ecl_lat,
            astrometric_params_solved,
            _warnings=warnings,
        )
        parallax_corrected = parallax - zpvals
        return parallax_corrected


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


def get_catalog(catalog_name: str) -> Catalog:
    """
    Método que devuelve el catálogo correspondiente.

    Parameters
    ----------
    catalog_name: str
        Nombre del catálogo

    Returns
    -------
    catalog: Catalog
        Catalogo correspondiente.
    """
    dic_catalog = {
        GaiaDR2.catalog_name: GaiaDR2,
        GaiaDR3.catalog_name: GaiaDR3,
        GaiaFPR.catalog_name: GaiaFPR,
    }
    try:
        return dic_catalog[catalog_name]()
    except KeyError:
        raise ValueError("El catalogo %s no está implementado." % catalog_name)
