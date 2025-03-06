import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

_ROOT = os.path.abspath(os.path.dirname(__file__))

# The file table_u0_g_c_p5.txt is a lookup table for the function u0(g,c) used
# to compute RUWE for five-parameter solutions (astrometric_params_solved = 31)
# according to
#
# RUWE = UWE / u0(G,C).
#
# Here,
#
# UWE = sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al-N)),
# N = 5,
# g = phot_g_mean_mag, and
# c = nu_eff_used_in_astrometry.
#
# Similarly, the file table_u0_g_c_p6.txt contains u0(g,c) for six-parameter solutions
# (astrometric_params_solved = 95), with notations as above except that N = 6
# and c = pseudocolour.


# n_p= N = Número de parámetros a ajustar (5 o 6).
# http://www.loganpearcescience.com/research/RUWE_as_an_indicator_of_multiplicity.pdf


def load_dataframe(csv_file: str) -> pd.DataFrame:
    """
    Función que carga los datos de csv_files dentro del mismo directorio de este código.

    Parameters
    ----------
    csv_file: str
        Ruta del archivo desde este directorio.

    Returns
    -------
    data: pd.DataFrame
        Datos cargados.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, csv_file)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"El archivo no se encuentra: {full_path}")

    data = pd.read_csv(full_path)
    return data


class U0Interpolator:
    """
    Class which holds functions that can used to calculate u0 values given the G-band magnitude
    only, or the G-band magnitude and the BP-RP colour of a source. The class initialization
    takes care of reading the necessary data and setting up the interpolators.
    """

    def __init__(self, n_p: int = 5):
        """
        Clase instanciadora.

        Parameters
        ----------
         n_p: int
            Número de parámetros, puede ser 5 o 6
        """
        self.n_p = n_p
        path = None
        if self.n_p == 5:
            path = "gaiadr3_u0_functions/table_u0_g_c_p5.txt"
        if self.n_p == 6:
            path = "gaiadr3_u0_functions/table_u0_g_c_p6.txt"

        if path is None:
            raise ValueError("n_p debe tomar un valor de 5 o 6")

        self.data = load_dataframe(path)

    def get_u0_g_c(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Función que dado el número de parámetros carga la función asociada a u0 según
        L. Lindegren (2023 Sep 13)

        Returns
        -------
        function: Callable[[np.ndarray, np.ndarray], np.ndarray]
            Función de dos dimensiones que dados g y c devuelve u0.

        Raises
        -------
        ValueError: Si n_p es distinto a 5 o 6.
        """

        df = self.data
        df.columns = df.columns.str.strip()
        g_mesh = df.g.unique()
        c_mesh = df.c.unique()
        u0_grid = df.u0.values.reshape((g_mesh.size, c_mesh.size))
        function = RectBivariateSpline(g_mesh, c_mesh, u0_grid)

        return function

    def get_u0(self, df) -> np.ndarray:
        """
        Función que calcula el factor de normalización u0 de RUWE.

        Parameters
        ----------
        df: pd.DataFrame
            Tabla descargada del catálogo de GAIA.

        Returns
        -------
        u0: np.ndarray
            Factor de normalización u0
        """
        spline = self.get_u0_g_c()
        g = df.phot_g_mean_mag.values
        c = df.nu_eff_used_in_astrometry.values
        u0 = np.array([spline(g[i], c[i])[0, 0] for i in range(g.size)])
        return u0

    def get_uwe_from_gaia(self, df: pd.DataFrame) -> np.ndarray:
        """
        Función que calcula el estadístico UWE según L. Lindegren (2023 Sep 13)

        Parameters
        ----------
        df: pd.DataFrame
            Tabla descargada del catálogo de GAIA.

        Returns
        -------
        uwe: np.ndarray
            Estadístico uwe para cada fila.
        """
        astrometric_chi2_al = df.astrometric_chi2_al.values
        astrometric_n_good_obs_al = df.astrometric_n_good_obs_al.values - self.n_p
        return np.sqrt(astrometric_chi2_al / astrometric_n_good_obs_al)

    def get_ruwe_from_gaia(self, df: pd.DataFrame) -> np.ndarray:
        """
        Función que calcula el estadístico RUWE según L. Lindegren (2023 Sep 13)

        Parameters
        ----------
        df: pd.DataFrame
            Tabla descargada del catálogo de GAIA.

        Returns
        -------
        uwe: np.ndarray
            Estadístico uwe para cada fila.
        """
        uwe = self.get_uwe_from_gaia(df)
        u0 = self.get_u0(df)
        ruwe = uwe / u0
        return ruwe
