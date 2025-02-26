import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

COLS = ["pmra", "pmdec", "parallax"]
COLS_CLUS = COLS + ["bp_rp", "phot_g_mean_mag"]


def get_main_cluster(labels: np.ndarray) -> int:
    """
    Función que calcula el cluster mayoritario
    """
    unique_lab = np.unique(labels[labels > -1])
    j = np.argmax(np.bincount(labels[labels > -1]))
    return unique_lab[j]


def score_cluster(df: pd.DataFrame, columns: list[str], labels: np.ndarray) -> np.ndarray:
    """
    Función que devuelve al suma de la dispersión de las columnas para cada cluster
    definido en labels
    """
    unique_lab = np.unique(labels[labels > -1])
    score = np.zeros(unique_lab.size)

    for i, lab in enumerate(unique_lab):
        gc = df[labels == lab]
        score[i] = gc[columns].describe().loc["std"].sum()
    return score


def clustering_dbscan(
    df_stars: pd.DataFrame, columns: list[str], columns_to_clus: list[str], max_cluster: int = 10
) -> np.ndarray:
    """
    Función que clusteriza los datos del catálogo usando las columnas columns_to_clus
    minimizando la desviación típica intercluster de las columns.

    Parameters
    ----------
    df_stars: pd.DataFrame,
        Tabla con las estrellas
    columns: list[str],
        Columnas a calcular la desviación típica.
    columns_to_clus: list[str]
        Columnas usadas en la clusterización.

    Returns
    -------
    labels: np.ndarray
        Array con las etiquetas.
    """

    data = StandardScaler().fit_transform(df_stars[columns_to_clus])
    mask_nan = df_stars[columns_to_clus].isna().any(axis=1).values
    df = df_stars[~mask_nan]
    data = data[~mask_nan]

    best_score = 1e6
    best_params = None

    for eps in np.linspace(0.1, 1.0, 10):
        for min_samples in range(3, 10):
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels = clustering.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if len(set(labels[labels > -1])) > 1 and n_clusters_ < max_cluster:
                score = np.median(score_cluster(df, columns, labels))
                if score < best_score:
                    best_score = score
                    best_params = (eps, min_samples)

    logging.info(
        f"Mejores parámetros encontrados: " f"eps={best_params[0]}, min_samples={best_params[1]}."
    )
    db = DBSCAN(eps=best_params[0], min_samples=best_params[1]).fit(data)
    return db.labels_


def cluster_representation_with_hvs(
    df_stars: pd.DataFrame,
    df_hvs_candidates: pd.DataFrame,
    labels: np.ndarray,
    factor_sigma: float = 2.0,
    hvs_pm: float = 50,
    df_source_x: Optional[pd.DataFrame] = None,
) -> tuple[plt.Figure, plt.Axes]:
    mask = labels == get_main_cluster(labels)
    df_gc = df_stars[mask]

    parallax_range = [
        df_gc.parallax.mean() - factor_sigma * df_gc.parallax.std(),
        df_gc.parallax.mean() + factor_sigma * df_gc.parallax.std(),
    ]

    mask_p = (df_hvs_candidates.parallax > parallax_range[0]) & (
        df_hvs_candidates.parallax < parallax_range[1]
    )

    mask_hvs = ((df_hvs_candidates.pm.abs() > hvs_pm)) & mask_p
    fig, ax = plt.subplots(figsize=(8, 5))

    selected = df_hvs_candidates[mask_hvs]
    factor = 200

    # Graficar las posiciones de las estrellas del cúmulo
    ax.scatter(df_gc.l, df_gc.b, s=1, color="grey", alpha=0.5)

    # Graficar vectores de movimiento propio
    ax.quiver(
        df_gc.l,
        df_gc.b,
        df_gc.pm_l / factor,
        df_gc.pm_b / factor,
        color="grey",
        scale=5,
        width=0.003,
    )

    # Marcar las estrellas seleccionadas (ejemplo: aquellas con ciertas condiciones)

    ax.quiver(
        selected["l"],
        selected["b"],
        selected["pm_l"] / factor,
        selected["pm_b"] / factor,
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
    ax.legend()
    return fig, ax
