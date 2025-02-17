from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = ["Zini", "MH", "logAge", "Mini","int_IMF", "Mass",   "logL", "logTe",  "logg",  "label",
           "McoreTP", "C_O",  "period0",  "period1",  "period2",  "period3",  "period4",  "pmode",
           "Mloss",  "tau1m",   "X",   "Y",   "Xc",  "Xn",  "Xo",  "Cexcess",  "Z", "mbolmag",
           "Gmag",    "G_BPmag",  "G_RPmag"]

def read_isochrone(pathfile_parsec: str) -> pd.DataFrame:
    """
    Función que lee los datos de una isochrona generada por PARSEC usando el catálogo
    GAIADR3 con all vega Magnitudes.

    Parameters
    ----------
    pathfile_parsec: src
        Ruta del archivo con la isochrona.

    Returns
    -------
    df_isc: pd.DataFrame
        Tabla con los datos de la Isochrona.
    """
    # Contar cuántas líneas de encabezado hay
    with open(pathfile_parsec, "r") as file:
        lines = file.readlines()

    # Encontrar la línea que contiene los nombres de las columnas
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            header_index = i - 1  # La línea anterior a los datos contiene los nombres de las columnas
            break

    # Cargar el archivo con pandas
    df_isc = pd.read_csv(
        pathfile_parsec,
        delim_whitespace=True,
        skiprows=header_index,
        comment='#',
        names=COLUMNS
    )

    return df_isc


def plot_cmd(
        df_r: pd.DataFrame,
        output_pathfile: str,
        pathfile_parsec: Optional[str] = None,
        distance_modulus: float = 0,
        redding: float = 0
) -> None:
    """
    Función que genera la gráfica con el CMD del catálgo df_r y de la isocrhona
    asociada si se le indica. El diagrama se genera sobre el color B-R y la magnitud vega
    G.

    Parameters
    ----------
    df_r: pd.DataFrame
        Catálogo a generar el diagrama color magnitud.
    output_pathfile: str
        Nombre de la ruta del archivo donde se quiere guardar.
    pathfile_parsec: Optional[str], default None
        Ruta del archivo de la isochrona. Si no se le indica no la pinta.
    distance_modulus: float, default 0
        Corrección de la magintud de la isochrona según el módulo de distancia.
    redding: float, default 0
        Corrección del color de la isochrona según el enrojecimiento sufrido por el objeto.
    """

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.scatter(x=df_r.bp_rp, y=df_r.phot_g_mean_mag, s=10, c="k", edgecolor='none',
                alpha=0.5)
    if pathfile_parsec:
        df_isc = read_isochrone(pathfile_parsec)
        plt.scatter(
            x=df_isc["G_BPmag"] - df_isc["G_RPmag"] + redding,
            y=df_isc["Gmag"] + distance_modulus,
            s=10,
            c="b",
            edgecolor='none',
            alpha=0.5
        )

    # Etiquetas de los ejes
    ax.set_xlabel('bp_rp')
    ax.set_ylabel('phot_g_mean_mag')

    # Ajustar límites de los ejes si es necesario
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 30)
    plt.gca().invert_yaxis()
    plt.savefig(output_pathfile)
    # Mostrar la gráfica
    plt.show()
