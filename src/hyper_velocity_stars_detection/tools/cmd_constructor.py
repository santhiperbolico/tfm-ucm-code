import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

COLUMNS_ISOCHRONE = [
    "Zini", "MH", "logAge", "Mini","int_IMF", "Mass",   "logL", "logTe",  "logg",  "label",
    "McoreTP", "C_O",  "period0",  "period1",  "period2",  "period3",  "period4",  "pmode",
    "Mloss",  "tau1m",   "X",   "Y",   "Xc",  "Xn",  "Xo",  "Cexcess",  "Z", 	 "mbolmag",
    "Gmag",    "G_BPmag",  "G_RPmag"]


def load_isochrone_from_parsec(
    file_path: str,
    columns: list[str] = COLUMNS_ISOCHRONE,
    isochrone_color_mag_1: str = "G_BPmag",
    isochrone_color_mag_2: str = "G_RPmag",
    isochrone_mag_y: str = "Gmag",
    color_field: str = "bp_rp",
    magnitud_field: str = "phot_g_mean_mag",
) -> pd.DataFrame:
    """
    Función que carga los datos de la isochrona generada por ParSec.

    Parameters
    ----------
    file_path: str
        Ruta del archivo.
    columns: list[str], default COLUMNS_ISOCHRONE
        Lista con las columnas del archivo de la isochrona.
    isochrone_color_mag_1: str
        Magnitud del espectro rojo del color
    isochrone_color_mag_2: str
        Magnitud del espectro azul del color.
    isochrone_mag_y: str
        Magnitud usada en el CMD
    color_field: str
        Nombre del campo de color del CMD.
    magnitud_field: str
        Nombre del campo de la magnitud.

    Returns
    -------
    df_isochrone: pd.DataFrame
        Tabla de datos cargados de la isochrona.
    """
    # Contar cuántas líneas de encabezado hay
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Encontrar la línea que contiene los nombres de las columnas
    header_index = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header_index = i - 1  # La línea anterior a los datos contiene los nombres de las columnas
            break

    # Cargar el archivo con pandas
    df_isochrone = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=header_index,
        comment='#',
        names=columns)

    isochrone_color = df_isochrone[isochrone_color_mag_1] - df_isochrone[isochrone_color_mag_2]
    df_isochrone[magnitud_field] = df_isochrone[isochrone_mag_y]
    df_isochrone[color_field] =isochrone_color
    return df_isochrone



def cmd_plot(
        df_catalog: pd.DataFrame,
        df_isochrone: Optional[pd.DataFrame] = None,
        color_field: str = "bp_rp",
        magnitud_field: str = "phot_g_mean_mag",
        isochrone_distance_module: float = 0,
        isochrone_redding: float = 0,
) -> tuple[plt.Axes, plt.Figure]:
    """
    Función que genera la gráfica del Color Magnitud Diagram. S
    Parameters
    ----------
    df_catalog: pd.DataFrame
        Tabla con las estrellas de la tabla.
    df_isochrone: pd.DataFrame
        Tabla con los datos de la isochrona.
    color_field: str
        Nombre del campo de color del CMD.
    magnitud_field: str
        Nombre del campo de la magnitud.
    isochrone_distance_module: float
        Módulo de distancia para la corrección de la isochrona.
    isochrone_redding: float
        Ajuste de enrojecimiento de la isochrona.

    Returns
    -------
    ax: Axes
        Eje de la gráfica.
    fig: Figure
        Figura con el CMD.
    """

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(15, 6))
    # Crear el scatter plot
    plt.scatter(
        x=df_catalog[color_field],
        y=df_catalog[magnitud_field],
        s=10,
        c="k",
        edgecolor='none',
        alpha=0.5
    )
    if isinstance(df_isochrone, pd.DataFrame):
        df_is_fit = fit_isochrone(
            df_isochrone, isochrone_distance_module,
            isochrone_redding, color_field, magnitud_field)
        plt.scatter(x=df_is_fit[color_field] , y=df_is_fit[magnitud_field],
                    s=10, c="b", edgecolor='none', alpha=0.5)

    # Etiquetas de los ejes
    ax.set_xlabel(color_field)
    ax.set_ylabel(magnitud_field)
    plt.gca().invert_yaxis()
    return ax, fig

def cmd_with_cluster(
        df_catalog: pd.DataFrame,
        labels: np.ndarray,
        df_isochrone: Optional[pd.DataFrame] = None,
        color_field: str = "bp_rp",
        magnitud_field: str = "phot_g_mean_mag",
        isochrone_distance_module: float = 0,
        isochrone_redding: float = 0,
) -> tuple[plt.Axes, plt.Figure]:
    """
    Función que genera la gráfica del Color Magnitud Diagram. S
    Parameters
    ----------
    df_catalog: pd.DataFrame
        Tabla con las estrellas de la tabla.
    labels: np.ndarray
        Array que etiqueta las estrellas según los clusters
    df_isochrone: pd.DataFrame
        Tabla con los datos de la isochrona.
    color_field: str
        Nombre del campo de color del CMD.
    magnitud_field: str
        Nombre del campo de la magnitud.
    isochrone_distance_module: float
        Módulo de distancia para la corrección de la isochrona.
    isochrone_redding: float
        Ajuste de enrojecimiento de la isochrona.

    Returns
    -------
    ax: Axes
        Eje de la gráfica.
    fig: Figure
        Figura con el CMD.
    """
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(15, 6))
    # Crear el scatter plot
    plt.scatter(
        x=df_catalog[color_field],
        y=df_catalog[magnitud_field],
        s=10,
        c="k",
        edgecolor='none',
        alpha=0.5
    )

    for label in np.unique(labels[labels > -1]):
        mask_i = labels == label
        plt.scatter(
            x=df_catalog.loc[mask_i, color_field],
            y=df_catalog.loc[mask_i, magnitud_field],
            s=10, edgecolor='none',alpha=0.4, label=f"cluster_{label}")

    if isinstance(df_isochrone, pd.DataFrame):
        df_is_fit = fit_isochrone(
            df_isochrone, isochrone_distance_module,
            isochrone_redding, color_field, magnitud_field)
        plt.scatter(x=df_is_fit[color_field], y=df_is_fit[magnitud_field],
                    s=10, c="b", edgecolor='none', alpha=0.5)

    # Etiquetas de los ejes
    ax.set_xlabel(color_field)
    ax.set_ylabel(magnitud_field)
    plt.gca().invert_yaxis()
    return ax, fig

# Función para aplicar el módulo de distancia y enrojecimiento a la isocrona
def fit_isochrone(
        isochrone: pd.DataFrame,
        distance_module: float,
        redding: float,
        color_field: str = "bp_rp",
        magnitud_field: str = "phot_g_mean_mag",
) -> pd.DataFrame:
    """
    Función que calcula los datos de la isochrona ajustando el módulo de distancia o el
    enrojecimiento.

    Parameters
    ----------
    isochrone: pd.DataFrame,
        Tabla de la Isochrona.
    distance_module: float,
        Módulo de la distancia.
    redding: float,
        Enrojecimiento.
    color_field: str = "bp_rp",
        Nombre del campo color.
    magnitud_field: str = "phot_g_mean_mag",
        Nombre del campo de la magnitud.

    Returns
    -------
    isocrone_fitted: pd.DataFrame
        Isochrona ajustada.

    """
    isocrone_fitted = isochrone.copy()
    isocrone_fitted[color_field] += redding
    isocrone_fitted[magnitud_field] += distance_module
    return isocrone_fitted

# Función de costo para minimizar la distancia entre la isocrona y las estrellas
def target_function(
        params: tuple[float, float],
        stars: pd.DataFrame,
        isochrone: pd.DataFrame,
        color_field: str = "bp_rp",
        magnitud_field: str = "phot_g_mean_mag",
):
    """
    Función que calcula la media de distancias de la isochrona ajustada al catálogo de estrellas
    asociadas.

    Parameters
    ----------
    params: tuple[float, float],
        Tupla con los parámetros de distance_module y redding
    stars: pd.DataFrame,
        Tabla con las estrellas del catálogo.
    isochrone: pd.DataFrame,
        Tabla con la isochrona original
    color_field: str = "bp_rp",
        Nombre del campo color.
    magnitud_field: str = "phot_g_mean_mag",
        Nombre del campo de la magnitud.

    Returns
    -------
    distance: float
        Distancia media de la isochrona.

    """
    distance_module, redding = params
    isocrone_fitted = fit_isochrone(isochrone, distance_module, redding)
    # Calcular la distancia cuadrática media entre las estrellas y la isocrona
    distances = np.sqrt(
        (stars[color_field].values[:, None] - isocrone_fitted[color_field].values[None, :]) ** 2 +
        (stars[magnitud_field].values[:, None] - isocrone_fitted[magnitud_field].values[None, :]
         ) ** 2)
    return np.nanmean(np.min(distances, axis=1))

def get_best_isochrone_fitted(
        stars: pd.DataFrame,
        isochrone: pd.DataFrame,
        color_field: str = "bp_rp",
        magnitud_field: str = "phot_g_mean_mag",
) -> tuple[float, float]:
    """

    Parameters
    ----------
    stars: pd.DataFrame,
        Tabla con las estrellas del catálogo.
    isochrone: pd.DataFrame,
        Tabla con la isochrona original
    color_field: str = "bp_rp",
        Nombre del campo color.
    magnitud_field: str = "phot_g_mean_mag",
        Nombre del campo de la magnitud.

    Returns
    -------
    distance_module: float
        Módulo de distancia ajustado
    redding: float
        Enrojecimiento.
    """

    # Optimización del módulo de distancia y enrojecimiento
    x0 = [10, 0.1]  # Ejemplo de valores iniciales
    resultado = minimize(
        target_function,
        x0,
        args=(stars, isochrone, color_field, magnitud_field),
        method='Nelder-Mead'
    )

    # Obtener los parámetros óptimos
    distance_module, redding = resultado.x
    logging.info(str(resultado))
    return distance_module, redding