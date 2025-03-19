import gzip
import logging
import os
import shutil
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from astroquery.esa.xmm_newton import XMMNewton
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import InvalidFileFormat, StorageObjectPandasCSV


@attrs
class XSource:
    """
    Elemento encargado en descargar y gestionar la fuente de rayos X desde el XMNNewton.
    """

    path = attrib(type=str, init=True)

    results = attrib(type=pd.DataFrame, init=False)
    obs_ids = attrib(type=list[str], init=False)

    @property
    def path_xsource(self) -> str:
        return os.path.join(self.path, "xsource")

    @property
    def path_lightcurves(self) -> str:
        return os.path.join(self.path_xsource, "light_curves")

    def download_data(self, ra: float, dec: float, radius: float) -> None:
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
        self.results = query_results[
            [col for col in query_results.columns if col not in discard_cols]
        ].to_pandas()

        os.makedirs(self.path_xsource, exist_ok=True)
        path_file = os.path.join(self.path_xsource, "xsource_data")
        StorageObjectPandasCSV().save(path_file, self.results)

    def download_light_curves(self, path: str):
        """
        Método que descarga las curvas de luz de las observaciones en rayos X asociadas
        a la fuente.

        Parameters
        ----------
        path: str
            Ruta donde se quieren guardar lso archivos FITS de las observaciones.
        """
        os.makedirs(self.path_lightcurves, exist_ok=True)
        self.obs_ids = self.results.observation_id.astype(str).unique().tolist()

        for obs_id in self.obs_ids:
            n_id = len(obs_id)
            if n_id < 10:
                obs_id = "".join(["0"] * int(10 - n_id)) + obs_id
            self._download_light_curve(obs_id)

    def _download_light_curve(self, obs_id: str) -> None:
        """
        Método que descarga las curvas de luz de la observación obs_id.
        Las observaciones se guardan en self.path_lightcurves/obs_id en formato
        FITS.

        Parameters
        ----------
        obs_id: str
            Id de la observación a descargar.
        """
        file_tar = f"xmm_data_{obs_id}.tar"
        XMMNewton.download_data(obs_id, extension="FTZ", filename=file_tar, cache=False)

        os.rename(file_tar, os.path.join(self.path_lightcurves, file_tar))
        file_tar = os.path.join(self.path_lightcurves, file_tar)

        extract_dir = os.path.join(self.path_lightcurves, obs_id)

        os.makedirs(extract_dir, exist_ok=True)

        dic_data = True
        iteration = 1
        while dic_data and iteration < 1000:
            dic = XMMNewton.get_epic_lightcurve(file_tar, iteration)
            dic_data = len(dic) > 0
            for key, fits_list in dic.items():
                for l_file in fits_list:
                    lc_ftz_file = l_file
                    lc_fits_file = lc_ftz_file.replace(".FTZ", ".FITS")
                    file = lc_fits_file.split("/")[-1]
                    with gzip.open(lc_ftz_file, "rb") as f_in:
                        with open(lc_fits_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(lc_ftz_file)
                    os.rename(lc_fits_file, os.path.join(extract_dir, file))
            iteration += 1
        try:
            os.removedirs(os.path.join(obs_id, "pps"))
        except FileNotFoundError:
            logging.info(f"No hay datos de la curva de luz de {obs_id}.")

    def save(self, path: Optional[str] = None) -> None:
        """
        Método que guarda los elementos de la fuente de rayos x en un archivo ZIP.

        Parameters
        ----------
        path: Optional[str], default None
            Ruta donde para guardar los datos. Por defecto se usa la establecida.
        """
        if path is None:
            path = self.path_xsource
        shutil.make_archive(path, "zip", self.path_xsource)

    def load(self, path: Optional[str] = None):
        """
        Método que descomprime los elementos de la fuente de rayos x desde un archivo ZIP.

        Parameters
        ----------
        path: Optional[str], default None
            Ruta donde para guardar los datos. Por defecto se usa la establecida.
        """
        path_zip = self.path_xsource + ".zip"

        if path is None:
            path = path_zip

        with ZipFile(path, "r") as zip_instance:
            if zip_instance.testzip() is not None:
                raise InvalidFileFormat(f"El archivo '{path}' no es un archivo zip válido")

            zip_instance.extractall(self.path_xsource)

        path_file = os.path.join(path, "xsource_data.csv")
        self.results = StorageObjectPandasCSV().load(path_file)
        return self.results
