import os.path

import pandas as pd
from astroquery.esa.xmm_newton import XMMNewton
from attr import attrib, attrs

from hyper_velocity_stars_detection.data_storage import StorageObjectPandasCSV


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
