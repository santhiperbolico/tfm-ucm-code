import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

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


def get_u0_g_c(n_p: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Función que dado el número de parámetros carga la función asociada a u0 según
    L. Lindegren (2023 Sep 13)

    Parameters
    ----------
    n_p: int,
        Número de parámetros, puede ser 5 o 6

    Returns
    -------
    function: Callable[[np.ndarray, np.ndarray], np.ndarray]
        Función de dos dimensiones que dados g y c devuelve u0.

    Raises
    -------
    ValueError: Si n_p es distinto a 5 o 6.
    """
    path = None
    if n_p == 5:
        path = "u0_functions/table_u0_g_c_p5.txt"
    if n_p == 6:
        path = "u0_functions/table_u0_g_c_p6.txt"

    if path is None:
        raise ValueError("n_p debe tomar un valor de 5 o 6")
    df = load_dataframe(path)
    df.columns = df.columns.str.strip()
    g_mesh = df.g.unique()
    c_mesh = df.c.unique()
    u0_grid = df.u0.values.reshape((g_mesh.size, c_mesh.size))
    function = RectBivariateSpline(g_mesh, c_mesh, u0_grid)

    return function


def get_uwe_from_gaia(df: pd.DataFrame, n_p: int) -> np.ndarray:
    """
    Función que calcula el estadístico UWE según L. Lindegren (2023 Sep 13)

    Parameters
    ----------
    df: pd.DataFrame
        Tabla descargada del catálogo de GAIA.
    n_p: int
        Número de parámetros.

    Returns
    -------
    uwe: np.ndarray
        Estadístico uwe para cada fila.
    """
    astrometric_chi2_al = df.astrometric_chi2_al.values
    astrometric_n_good_obs_al = df.astrometric_n_good_obs_al.values - n_p
    return np.sqrt(astrometric_chi2_al / astrometric_n_good_obs_al)


def get_ruwe_from_gaia(df: pd.DataFrame, n_p: int) -> np.ndarray:
    """
    Función que calcula el estadístico RUWE según L. Lindegren (2023 Sep 13)

    Parameters
    ----------
    df: pd.DataFrame
        Tabla descargada del catálogo de GAIA.
    n_p: int
        Número de parámetros.

    Returns
    -------
    uwe: np.ndarray
        Estadístico uwe para cada fila.
    """
    uwe = get_uwe_from_gaia(df, n_p)
    g = df.phot_g_mean_mag.values
    c = df.nu_eff_used_in_astrometry.values
    spline = get_u0_g_c(n_p)
    ruwe = np.zeros(uwe.size)
    for i in range(uwe.size):
        uwe0 = spline(g[i], c[i])[0, 0]
        ruwe[i] = uwe[i] / uwe0
    return ruwe
