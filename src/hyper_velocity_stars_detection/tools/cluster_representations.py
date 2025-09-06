import logging
from typing import Optional

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hyper_velocity_stars_detection.variables_names import (
    BP_RP,
    G_LATITUDE,
    G_LONGITUDE,
    G_MAG,
    PM_G_LATITUDE,
    PM_G_LONGITUDE,
)

COLUMNS_ISOCHRONE = [
    "Zini",
    "MH",
    "logAge",
    "Mini",
    "int_IMF",
    "Mass",
    "logL",
    "logTe",
    "logg",
    "label",
    "McoreTP",
    "C_O",
    "period0",
    "period1",
    "period2",
    "period3",
    "period4",
    "pmode",
    "Mloss",
    "tau1m",
    "X",
    "Y",
    "Xc",
    "Xn",
    "Xo",
    "Cexcess",
    "Z",
    "mbolmag",
    "Gmag",
    "G_BPmag",
    "G_RPmag",
]


def load_isochrone_from_parsec(
    file_path: str,
    columns: Optional[list[str]] = None,
    isochrone_color_mag_1: str = "G_BPmag",
    isochrone_color_mag_2: str = "G_RPmag",
    isochrone_mag_y: str = "Gmag",
    color_field: str = BP_RP,
    magnitud_field: str = G_MAG,
) -> pd.DataFrame:
    """
    Función que carga los datos de la isochrona generada por ParSec.

    Parameters
    ----------
    file_path: str
        Ruta del archivo.
    columns: Optional[list[str]] = None,
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
    if columns is None:
        columns = COLUMNS_ISOCHRONE
    # Contar cuántas líneas de encabezado hay
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Encontrar la línea que contiene los nombres de las columnas
    header_index = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header_index = (
                i - 1
            )  # La línea anterior a los datos contiene los nombres de las columnas
            break

    # Cargar el archivo con pandas
    df_isochrone = pd.read_csv(
        file_path, delim_whitespace=True, skiprows=header_index, comment="#", names=columns
    )

    isochrone_color = df_isochrone[isochrone_color_mag_1] - df_isochrone[isochrone_color_mag_2]
    df_isochrone[magnitud_field] = df_isochrone[isochrone_mag_y]
    df_isochrone[color_field] = isochrone_color
    return df_isochrone


def cmd_plot(
    df_catalog: pd.DataFrame,
    df_isochrone: Optional[pd.DataFrame] = None,
    color_field: str = BP_RP,
    mag_field: str = G_MAG,
    isochrone_distance_module: float = 0,
    isochrone_redding: float = 0,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[int, int] = (15, 6),
) -> tuple[plt.Figure, plt.Axes]:
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
    mag_field: str
        Nombre del campo de la magnitud.
    isochrone_distance_module: float
        Módulo de distancia para la corrección de la isochrona.
    isochrone_redding: float
        Ajuste de enrojecimiento de la isochrona.

    Returns
    -------
    fig: Figure
        Figura con el CMD.
    ax: Axes
        Eje de la gráfica.
    """

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Crear el scatter plot
    plt.scatter(
        x=df_catalog[color_field],
        y=df_catalog[mag_field],
        s=10,
        c="k",
        edgecolor="none",
        alpha=0.5,
    )
    if isinstance(df_isochrone, pd.DataFrame):
        df_is_fit = fit_isochrone(
            df_isochrone, isochrone_distance_module, isochrone_redding, color_field, mag_field
        )
        plt.scatter(
            x=df_is_fit[color_field],
            y=df_is_fit[mag_field],
            s=10,
            c="b",
            edgecolor="none",
            alpha=0.5,
        )

    # Etiquetas de los ejes
    ax.set_xlabel(color_field)
    ax.set_ylabel(mag_field)
    plt.gca().invert_yaxis()
    return fig, ax


def cmd_with_cluster(
    df_catalog: pd.DataFrame,
    labels: np.ndarray,
    df_isochrone: Optional[pd.DataFrame] = None,
    color_field: str = BP_RP,
    mag_field: str = G_MAG,
    isochrone_distance_module: float = 0,
    isochrone_redding: float = 0,
    clusters: Optional[int | list[int]] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = False,
    figsize: tuple[int, int] = (15, 6),
) -> tuple[plt.Figure, plt.Axes]:
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
    mag_field: str
        Nombre del campo de la magnitud.
    isochrone_distance_module: float
        Módulo de distancia para la corrección de la isochrona.
    isochrone_redding: float
        Ajuste de enrojecimiento de la isochrona.
    legend: bool, default False
            Indica si se quiere graficar la leyenda.

    Returns
    -------
    fig: Figure
            Figura con el CMD.
    ax: Axes
        Eje de la gráfica.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        x=df_catalog[color_field],
        y=df_catalog[mag_field],
        s=10,
        c="k",
        edgecolor="none",
        alpha=0.5,
    )
    if clusters is None:
        clusters = np.unique(labels[labels > -1]).tolist()

    if isinstance(clusters, int):
        clusters = [clusters]

    for label in clusters:
        mask_i = labels == label
        ax.scatter(
            x=df_catalog.loc[mask_i, color_field],
            y=df_catalog.loc[mask_i, mag_field],
            s=10,
            edgecolor="none",
            alpha=0.4,
            label=f"cluster_{label}",
        )

    if isinstance(df_isochrone, pd.DataFrame):
        df_is_fit = fit_isochrone(
            df_isochrone, isochrone_distance_module, isochrone_redding, color_field, mag_field
        )
        ax.scatter(
            x=df_is_fit[color_field],
            y=df_is_fit[mag_field],
            s=15,
            edgecolor="none",
            label="Isocrona",
        )

    # Etiquetas de los ejes
    ax.set_xlabel(color_field)
    ax.set_ylabel(mag_field)
    ax.invert_yaxis()
    if legend:
        ax.legend()
    return fig, ax


# Función para aplicar el módulo de distancia y enrojecimiento a la isocrona
def fit_isochrone(
    isochrone: pd.DataFrame,
    distance_module: float,
    redding: float,
    color_field: str = BP_RP,
    magnitud_field: str = G_MAG,
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
    color_field: str = BP_RP,
    magnitud_field: str = G_MAG,
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
        (stars[color_field].values[:, None] - isocrone_fitted[color_field].values[None, :]) ** 2
        + (stars[magnitud_field].values[:, None] - isocrone_fitted[magnitud_field].values[None, :])
        ** 2
    )
    return np.nanmean(np.min(distances, axis=1))


def get_best_isochrone_fitted(
    stars: pd.DataFrame,
    isochrone: pd.DataFrame,
    color_field: str = BP_RP,
    magnitud_field: str = G_MAG,
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
        method="Nelder-Mead",
    )

    # Obtener los parámetros óptimos
    distance_module, redding = resultado.x
    logging.info(str(resultado))
    return distance_module, redding


def cluster_representation_with_hvs(
    df_gc: pd.DataFrame,
    df_hvs_candidates: pd.DataFrame,
    factor_sigma: float = 1.0,
    factor_size: int = 200,
    hvs_pm: float = 150,
    df_source_x: Optional[pd.DataFrame] = None,
    legend: bool = True,
    parallax_corrected: bool = True,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Función que representa el cluster con las candidatas HVS en coordenadas galacticas
    y con los vectores de proper motion.

    Parameters
    ----------
    df_gc: pd.DataFrame
        Catalogo de estrellas del cluster
    df_hvs_candidates: pd.DataFrame
        Catálogo de estrellas donde se quiere buscar las HVS
    factor_sigma: float, default 1
        Proporción del sigma del paralaje que se quiere usar para seleccionar las HVS
    hvs_pm: float, default
        Movimiento propio mínimo en km por segundo en la selección de HVS
    df_source_x: Optional[pd.DataFrame], None
        Si se indica tabla con las fuentes de rayos X a representar.
    legend: bool, default True
            Indica si se quiere graficar la leyenda.

    Returns
    -------
    fig: Figure
        Figura con la representación en coordenadas galactics
    ax: Axes
        Eje de la figura.
    """
    parallax_col = "parallax_corrected" if parallax_corrected else "parallax"

    parallax_range = [
        df_gc[parallax_col].mean() - factor_sigma * df_gc[parallax_col].std(),
        df_gc[parallax_col].mean() + factor_sigma * df_gc[parallax_col].std(),
    ]

    mask_p = (df_hvs_candidates[parallax_col] > parallax_range[0]) & (
        df_hvs_candidates[parallax_col] < parallax_range[1]
    )

    pm_candidates = df_hvs_candidates.pm_kms - df_gc.pm_kms.mean()
    mask_hvs = (pm_candidates > hvs_pm) & mask_p
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    selected = df_hvs_candidates[mask_hvs]

    mean_pm_l = df_gc.pm_l.mean()
    mean_pm_b = df_gc.pm_b.mean()

    # Graficar las posiciones de las estrellas del cúmulo
    ax.scatter(df_gc.l, df_gc.b, s=1, color="grey", alpha=0.5)

    # Graficar vectores de movimiento propio
    ax.quiver(
        df_gc.l,
        df_gc.b,
        (df_gc.pm_l - mean_pm_l) / factor_size,
        (df_gc.pm_b - mean_pm_b) / factor_size,
        color="grey",
        scale=5,
        width=0.003,
    )

    # Marcar las estrellas seleccionadas (ejemplo: aquellas con ciertas condiciones)
    if not selected.empty:
        ax.quiver(
            selected["l"],
            selected["b"],
            (selected["pm_l"] - mean_pm_l) / factor_size,
            (selected["pm_b"] - mean_pm_b) / factor_size,
            color="blue",
            scale=5,
            width=0.003,
            label="Pre-selected Stars",
        )

    if isinstance(df_source_x, pd.DataFrame):
        ax.scatter(
            df_source_x.lii.values,
            df_source_x.bii.values,
            marker="s",
            s=20,
            color="k",
            label="XR_Source",
        )

    # Etiquetas y detalles
    ax.set_xlabel("l (Galactic Longitude)")
    ax.set_ylabel("b (Galactic Latitude)")
    if legend:
        ax.legend()
    return fig, ax


def cluster_representation(
    df_gc: pd.DataFrame,
    df_highlights_stars: Optional[pd.DataFrame] = None,
    df_source_x: Optional[pd.DataFrame] = None,
    factor_size: int = 200,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    figsize: tuple[int, int] = (8, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Función que representa el cluster en coordenadas galacticas y con los vectores
    de proper motion. Si se le indica pinta la fuente de rayos X y estrellas a destacar.

    Parameters
    ----------
    df_gc: pd.DataFrame
        Catalogo de estrellas del cluster
    df_highlights_stars: pd.DataFrame
        Catálogo de estrellas que se quieren destacar
    df_source_x: Optional[pd.DataFrame], None
        Si se indica tabla con las fuentes de rayos X a representar.
    factor_size: int = 200,
        Controla el tamaño de los vectores.
    ax: Optional[plt.Axes] = None,
        Axis donde se quiere pintar la gráfica.
    legend: bool, default True
            Indica si se quiere graficar la leyenda.

    Returns
    -------
    fig: Figure
        Figura con la representación en coordenadas galactics
    ax: Axes
        Eje de la figura.
    """
    if not isinstance(df_highlights_stars, pd.DataFrame):
        df_highlights_stars = pd.DataFrame()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        is_wcs = False
    else:
        # Heurística sencilla para saber si es un eje WCS
        is_wcs = hasattr(ax, "get_transform") and hasattr(ax, "coords")

    tr = ax.transData
    if is_wcs:
        tr = ax.get_transform("galactic")
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_format_unit(u.deg, decimal=True)
        lat.set_format_unit(u.deg, decimal=True)
        lon.set_axislabel("l (Galactic Longitude)")
        lat.set_axislabel("b (Galactic Latitude)")

    mean_pm_l = df_gc[PM_G_LONGITUDE].mean()
    mean_pm_b = df_gc[PM_G_LATITUDE].mean()

    ax.scatter(
        df_gc[G_LONGITUDE] * u.deg,
        df_gc[G_LATITUDE] * u.deg,
        s=1,
        color="grey",
        alpha=0.5,
        transform=tr,
    )
    ax.quiver(
        df_gc[G_LONGITUDE] * u.deg,
        df_gc[G_LATITUDE] * u.deg,
        (df_gc[PM_G_LONGITUDE] - mean_pm_l) / factor_size,
        (df_gc[PM_G_LATITUDE] - mean_pm_b) / factor_size,
        color="grey",
        scale=5,
        width=0.003,
        transform=tr,
    )

    if not df_highlights_stars.empty:
        ax.quiver(
            df_highlights_stars[G_LONGITUDE] * u.deg,
            df_highlights_stars[G_LATITUDE] * u.deg,
            (df_highlights_stars[PM_G_LONGITUDE] - mean_pm_l) / factor_size,
            (df_highlights_stars[PM_G_LATITUDE] - mean_pm_b) / factor_size,
            color="blue",
            scale=5,
            width=0.003,
            label="Pre-selected Stars",
            transform=tr,
        )

    if isinstance(df_source_x, pd.DataFrame):
        ax.scatter(
            df_source_x.lii.values * u.deg,
            df_source_x.bii.values * u.deg,
            marker="s",
            s=20,
            color="k",
            label="XR_Source",
            transform=tr,
        )

    if not is_wcs:
        ax.set_xlabel("l (Galactic Longitude)")
        ax.set_ylabel("b (Galactic Latitude)")
    if legend:
        ax.legend()
    return fig, ax
