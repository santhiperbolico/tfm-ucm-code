import logging
from typing import Optional

import pandas as pd
from astroquery.gaia import Gaia
from astropy.table import Table

from attr import attrs, attrib


class CatalogsType:
    GAIA_DR2 = "gaiadr2"
    GAIA_DR3 = "gaiadr3"

class CatalogsTables:
    GAIA_DR2 = "gaiadr2.gaia_source"
    GAIA_DR3 = "gaiadr3.gaia_source"

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
            CatalogsType.GAIA_DR3: CatalogsTables.GAIA_DR3
        }
        try:
            catalog_table = dic_types[catalog_name]
        except KeyError:
            raise CatalogError(f"El catálogo {name} no existe.")
        return cls(catalog_table)

    def download_results(
            self,
            ra: float,
            dec: float,
            radius: float,
            output_file: str,
            filter_parallax_min: Optional[float]=None,
            filter_parallax_max: Optional[float]=None,
            filter_parallax_error: Optional[float]=None
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

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.
        """

        filter = f"""
        WHERE 1=CONTAINS(
               POINT('ICRS', ra, dec),
               CIRCLE('ICRS', {ra}, {dec}, {radius * 1})
               )
        """
        if filter_parallax_max:
            filter = f"{filter} AND parallax < {filter_parallax_max}"
        if filter_parallax_min:
            filter = f"{filter} AND parallax > {filter_parallax_min}"
        if filter_parallax_error:
            filter = f"{filter} AND parallax_error < {filter_parallax_error}"

        query = f"SELECT * FROM {self.catalog_table} {filter}"

        job = Gaia.launch_job_async(query, dump_to_file=True, output_format='votable',
                                    output_file=f"{output_file}")
        logging.info(job)
        results = job.get_results()
        return results.to_pandas()

    @staticmethod
    def read_catalog(file_catalog: str) -> pd.DataFrame:
        """
        Método que lee el archivo asociado al catálogo y lo convierte en DataFrame

        Parameters
        ----------
        file_catalog

        Returns
        -------
        results: pd.DataFrame
            Tabla con los datos del catálogo descargados.

        """
        results = Table.read(file_catalog, format="votable").to_pandas()
        return results