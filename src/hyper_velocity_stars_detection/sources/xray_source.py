import logging
import os
import shutil
from typing import Optional
from zipfile import ZipFile

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.heasarc import Heasarc
from astroquery.simbad import Simbad
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import InvalidFileFormat, StorageObjectPandasCSV
from hyper_velocity_stars_detection.sources.lightcurves import LightCurve

Simbad.TIMEOUT = 300
Simbad.ROW_LIMIT = 1


def get_main_id(name: str) -> str | None:
    """
    Función que extrae un  identificador único del objeto. Si no lo encuentra devuelve None.

    Parameters
    ----------
    name: str
        Nombre a buscar.

    Returns
    -------
    main_id: str | None
        Devuelve el identificador si lo encuentra
    """
    try:
        result = Simbad.query_object(name)
        if result:
            main_id = "_".join(result["MAIN_ID"][0].split())
            return main_id
    except Exception as e:
        logging.info(f"Error con {name}: {e}")
    return None


def get_obs_id(obs_ids: list[str | int]) -> list[str]:
    """
    Función que formatea los obs_id de XMMNewton.
    """
    list_ids = []
    for obs_id in obs_ids:
        obs_id = str(obs_id)
        n_id = len(obs_id)
        if n_id < 10:
            obs_id = "".join(["0"] * int(10 - n_id)) + obs_id
        list_ids.append(obs_id)
    return list_ids


@attrs(auto_attribs=True)
class XCatalog:
    mission: str
    searched_columns: list[str]

    format_columns = {
        "obsid": "str",
        "name": "str",
        "ra": "float",
        "dec": "float",
        "lii": "float",
        "bii": "float",
        "time": "float",
        "exposure": "float",
        "public_date": "float",
        "class": "str",
    }

    def download_data(self, coords: SkyCoord, radius: float) -> pd.DataFrame:
        """
        Método que descarga los datos del catalogo asociado teniendo en cuenta los parámetros.

        Parameters
        ----------
        coords: SkyCoord
            Coordenadas del objeto a descarar
        radius: float
            Radio de búsqueda en grados.
        """
        query_res = Heasarc.query_region(
            coords,
            mission=self.mission,
            radius=radius * u.deg,
            fields=", ".join(self.searched_columns),
        )
        results = query_res.to_pandas()
        object_cols = results.select_dtypes([object]).columns
        results[object_cols] = results[object_cols].astype(str)

        results = results.rename(
            columns=dict(zip(self.searched_columns, list(self.format_columns.keys())))
        )
        if results.empty:
            results = pd.DataFrame(columns=list(self.format_columns.keys()))

        for column, col_type in self.format_columns.items():
            if results[column].dtype == np.dtype("O"):
                results[column] = results[column].astype("str").str.strip()
                results.loc[results[column] == "", column] = None

            results[column] = results[column].astype(col_type)

        results.insert(0, "mission", self.mission)
        results.insert(1, "main_id", results.name.apply(get_main_id))
        return results


class XCatalogParams:
    """
    Clase que recoge el nombre de la misión y las columnas a extraer usando Heasarc.
    """

    XMNNEWTON = (
        "xmmmaster",
        [
            "OBSID",
            "NAME",
            "RA",
            "DEC",
            "LII",
            "BII",
            "TIME",
            "ESTIMATED_EXPOSURE",
            "PUBLIC_DATE",
            "CLASS",
        ],
    )
    CHANDRA = (
        "chanmaster",
        ["OBSID", "NAME", "RA", "DEC", "LII", "BII", "TIME", "EXPOSURE", "PUBLIC_DATE", "CLASS"],
    )


@attrs
class XSource:
    """
    Elemento encargado en descargar y gestionar la fuente de rayos X desde el XMNNewton.
    """

    path = attrib(type=str, init=True)

    results = attrib(type=pd.DataFrame, init=False)
    lightcurves = attrib(type=LightCurve, init=False)

    catalogs = [XCatalog(*XCatalogParams.XMNNEWTON), XCatalog(*XCatalogParams.CHANDRA)]

    def __attrs_post_init__(self):
        self.lightcurves = LightCurve(os.path.join(self.path_xsource, "light_curves"))

    @property
    def path_xsource(self) -> str:
        return os.path.join(self.path, "xsource")

    def download_data(self, coords: SkyCoord, radius: float) -> None:
        """
        Método que descarga los datos del catalogo asociado teniendo en cuenta los parámetros.

        Parameters
        ----------
        coords: SkyCoord
            Coordenadas del objeto a descarar
        radius: float
            Radio de búsqueda en grados.
        """
        self.results = pd.DataFrame()
        for catalog in self.catalogs:
            df_c = catalog.download_data(coords, radius)
            self.results = pd.concat((self.results, df_c))

    def download_light_curves(self, cache: bool = False):
        """
        Método que descarga las curvas de luz de las observaciones en rayos X asociadas
        a la fuente.

        Parameters
        ----------
        cache: bool, default False
            Indica si se quiere descargar usando la cache
        """

        # Filtramos por XMMNewton porque las curvas de luz solo se descargan del XMMNewton.
        xmm_results = self.results[self.results.mission == XCatalogParams.XMNNEWTON[0]]
        os.makedirs(self.lightcurves.path_lightcurves, exist_ok=True)
        obs_ids = get_obs_id(xmm_results.obsid.astype(str).unique().tolist())

        for obs_id in obs_ids:
            logging.info(f"Descargando observación {obs_id}")
            self.lightcurves.download_light_curve(obs_id, cache)

    def save(self, path: Optional[str] = None) -> None:
        """
        Método que guarda los elementos de la fuente de rayos x en un archivo ZIP.

        Parameters
        ----------
        path: Optional[str], default None
            Ruta donde para guardar los datos. Por defecto se usa la establecida.
        """
        if path is None:
            path = self.path
        path_xsource = os.path.join(path, "xsource")
        os.makedirs(self.path_xsource, exist_ok=True)
        path_file = os.path.join(self.path_xsource, "xsource_data")
        StorageObjectPandasCSV().save(path_file, self.results)
        shutil.make_archive(path_xsource, "zip", self.path_xsource)

    def load(self, path: Optional[str] = None):
        """
        Método que descomprime los elementos de la fuente de rayos x desde un archivo ZIP.

        Parameters
        ----------
        path: Optional[str], default None
            Ruta donde para guardar los datos. Por defecto se usa la establecida.
        """
        if path is None:
            path = self.path
        path_zip = os.path.join(path, "xsource.zip")

        with ZipFile(path_zip, "r") as zip_instance:
            if zip_instance.testzip() is not None:
                raise InvalidFileFormat(f"El archivo '{path_zip}' no es un archivo zip válido")
            zip_instance.extractall(self.path_xsource)
        path_file = os.path.join(self.path_xsource, "xsource_data.csv")
        self.results = StorageObjectPandasCSV().load(path_file)
        return self.results
