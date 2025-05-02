import pandas as pd
from astroquery.vizier import Vizier


def get_clusters_dr2() -> pd.DataFrame:
    """
    Función que descarga los datos de clusters del catálogo J/MNRAS/482/5138 extraidos
    del DR2 de Gaia.

    Returns
    -------
    clusters_dr2: pd.DataFrame
        Tabla con el catalogo del DR2 con la información de cada cluster:
            - Name: Nombre del cluster
            - parallax: Paralaje del cluster calculado como 1/Rsun.
            - e_parallax: Error en el paralaje estimado usando Rsun y e_Rsun.
            - Rsun: Distancia con respecto del en kpc
            - e_Rsun: Error de la distancia.
            - pmRA_: Movimiento propio en ascensión recta proyectado.
            - e_pmRA_: Error en el pmRA_.
            - pmDE: Movimiento propio en declinaje.
            - e_pmDE: Error en el pmDE.
            - RV: Velocidad radial
            - e_RV: Error en la velocidad radial.
            - rho: Correlación entre pmRA_ y pmDE.

    """
    v = Vizier(
        columns=[
            "Name",
            "Rsun",
            "e_Rsun",
            "pmRA*",
            "e_pmRA*",
            "pmDE",
            "e_pmDE",
            "RV",
            "e_RV",
            "rho",
        ]
    )
    v.ROW_LIMIT = -1
    catalogs = v.get_catalogs("J/MNRAS/482/5138")
    clusters_dr2 = catalogs[0].to_pandas()
    clusters_dr2.Name = clusters_dr2.Name.str.lower()

    clusters_dr2["parallax"] = 1 / (clusters_dr2.Rsun)
    clusters_dr2["e_parallax"] = (
        1 / (clusters_dr2.Rsun - clusters_dr2.e_Rsun)
        - 1 / (clusters_dr2.Rsun + clusters_dr2.e_Rsun)
    ) / 2
    clusters_dr2 = clusters_dr2[
        [
            "Name",
            "parallax",
            "e_parallax",
            "pmRA_",
            "e_pmRA_",
            "pmDE",
            "e_pmDE",
            "RV",
            "e_RV",
            "rho",
            "Rsun",
            "e_Rsun",
        ]
    ]
    return clusters_dr2


def get_mass_luminosity_clusters() -> pd.DataFrame:
    """
    Función que descarga lso datos del catálogo J/ApJS/161/304
    con las estimaciones de masa y luminosidad de clusters globulares.

    Returns
    -------
    clusters_ml: Catalogo con lso datos de masa, luminosidad y ratio de masa luminosidad.
        - SName: Nombre del cluster según SIMBAD.
        - Mtot: Logaritmo de la masa del cluster en masas solares.
        - e_Mtot: Error en la estimación de la masa.
        - Ltot: Logaritmo de la luminosidad del cluster en luminosidad solar.
        - e_Ltot: Error en la estimación de la luminosidad.
        - M_L: Ratio de Masa - Luminosidad_V sintetizada.
        - e_M_L: Error en el ratio de Masa - Luminosidad_V.
    """
    catalog1 = "J/ApJS/161/304/clusters"
    catalog2 = "J/ApJS/161/304/models"

    v = Vizier(columns=["**"])
    v.ROW_LIMIT = -1
    catalogs = v.get_catalogs([catalog1, catalog2])
    clusters_ml = catalogs[0].to_pandas()
    tbl_2 = catalogs[1].to_pandas()

    properties = ["Ltot", "Mtot", "M_L"]
    for prop in properties:
        df_p = tbl_2[["Cluster", prop, f"e_{prop}", f"E_{prop}"]].sort_values(
            ["Cluster", f"e_{prop}", f"E_{prop}"]
        )
        clusters_ml = pd.merge(
            clusters_ml, df_p.groupby("Cluster").first().reset_index(), on="Cluster"
        )

    clusters_ml.SName = clusters_ml.SName.str.lower()
    return clusters_ml


def get_all_cluster_data() -> pd.DataFrame:
    """
    Función que devuelve los datos de loc cumulos globulares extraidos por el DR2
    y por las estimaciones de Masa - Luminosidad_V.

    Returns
    -------
    df_clusters: pd.DataFrame
        Datos de los cúmulos globulares extraidos por el DR2 y por las estimaciones
        de Masa - Luminosidad_V.
    """
    clusters_dr2 = get_clusters_dr2()
    clusters_ml = get_mass_luminosity_clusters()
    df_clusters = pd.merge(
        clusters_dr2, clusters_ml, left_on=["Name"], right_on="SName", how="left"
    )
    return df_clusters
