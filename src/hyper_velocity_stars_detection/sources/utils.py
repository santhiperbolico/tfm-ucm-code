import logging
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astropy.units import Unit
from astroquery.gaia import Gaia
from astroquery.heasarc import Heasarc
from astroquery.simbad import Simbad
from zero_point import zpt

HEASARC_COLUMNS = [
    "NAME",
    "RA",
    "DEC",
    "CORE_RADIUS",
    "HALF_LIGHT_RADIUS",
    "CENTRAL_CONCENTRATION",
]


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


def get_skycoords(
    result: Table,
    unit_ra: Unit,
    unit_dec: Unit,
) -> SkyCoord:
    """
    Función que formatea el primer elemento de una tabla y devuelve sus coordenadas en un
    objeto SkyCoords.

    Parameters
    ----------
    result: Table
        Tabla donde el primer elemento es el que queremos extraer sus coordenadas.
    unit_ra: Unit
        Unidades de RA
    unit_dec: Unit
        Unidades de DEC

    Returns
    -------
    coords: SkyCoords
        Coordenadas.
    """
    ra = result["RA"][0]
    dec = result["DEC"][0]
    coords = SkyCoord(ra, dec, unit=(unit_ra, unit_dec), frame="icrs")
    logging.info(f"Coordenadas de RA = {ra}, DEC = {dec}")
    return coords


class DidntFindObject(Exception):
    """
    Excepción que salta cuando no se encuentra un objeto en un catálogo.
    """

    pass


def get_object_from_simbad(name: str, timeout: int = 120) -> Table:
    """
    Función que dado el nombre de un objeto lo busca en SIMBAD.

    Parameters
    ----------
    name: str
        Nombre del objeto en el catalgo de SIMBAD
    timeout: int, default 120
        Tiempo de esperar hasta encontrar el objeto en el catálogo de SIMBAD

    Return
    -------
    results: Table
        Tabla con la información de los objetos encontrados en el catálogo de SIMBAD

     Raises
     ------
     DidntFindObject: Cuando no se ecnuentra el objeto en el catálogo.
    """
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields("dim")
    custom_simbad.TIMEOUT = timeout
    results = custom_simbad.query_object(name)
    if results is None:
        raise DidntFindObject(f"No se encontró el objeto {name} en Simbad.")
    coords = SkyCoord(ra=results["RA"], dec=results["DEC"], unit=(u.hourangle, u.deg))

    results["RA"] = coords.ra.deg * u.deg
    results["DEC"] = coords.dec.deg * u.deg
    results["ANGULAR_SIZE"] = results["GALDIM_MAJAXIS"].value * u.arcmin
    results["RA"].format = "{:7.5f}"
    results["DEC"].format = "{:7.5f}"
    return results


def get_object_from_heasarc(name: str) -> Table:
    """
    Función que dado el nombre de un cluster lo busca en Heasarc GlobularCluster.

    Parameters
    ----------
    name: str
        Nombre del objeto en el catalgo de Heasarc

    Return
    -------
    results: Table
        Tabla con la información de los objetos encontrados en el catálogo de Heasarc

     Raises
     ------
     DidntFindObject: Cuando no se ecnuentra el objeto en el catálogo.
    """
    heasarc = Heasarc()
    mission = "globclust"
    result_cluster = heasarc.query_object(
        name, mission=mission, resultmax=10, fields=",".join(HEASARC_COLUMNS)
    )
    if result_cluster is None:
        raise DidntFindObject(f"No se encontró el objeto {name} en Heasarc.")

    result_cluster["ANGULAR_SIZE"] = (
        np.power(10, result_cluster["CENTRAL_CONCENTRATION"].value)
        * result_cluster["CORE_RADIUS"].value
    ) * u.arcmin
    return result_cluster


def get_object(name: str, radius_arcsec: float = 5.0, timeout: int = 120) -> Table:
    """
    Función que dado el nombre de un objeto lo busca en SIMBAD y
    realiza la búsqueda con un radio radius_arcsec en el catalogo de GAIA DR3 centrado
    en las coordenadas del objeto.

    Parameters
    ----------
    name: str
        Nombre del objeto en el catalgo de SIMBAD
    radius_arcsec: float, default 5
        Radio en arcosegundos
    timeout: int, default 120
        Tiempo de esperar hasta encontrar el objeto en el catálogo de SIMBAD

    Return
    -------
    results: Table
        Tabla con la información de los objetos encontrados en el catálogo de GAIA DR3
    """
    result_simbad = get_object_from_simbad(name, timeout)
    coords = get_skycoords(result_simbad, u.hourangle, u.deg)

    radius = radius_arcsec * u.arcsec
    job = Gaia.cone_search_async(coordinate=coords, radius=radius)
    results = job.get_results()
    logging.info(f"Se encontraron {len(results)} fuentes en el radio de búsqueda:")
    return results


def fix_parallax(df_data: pd.DataFrame, warnings: bool = True) -> pd.DataFrame:
    """
    Función que corrige el sesgo del paralaje de los datos del objeto calculando el punto zero
    del paralaje corregido (Lindegren et al., 2021 https://doi.org/10.1051/0004-6361/202039653).

    Parameters
    ----------
    df_data: pd.DataFrame
        Datos con las métricas
    warnings: bool, default True
        Indica si se quiere mostrar los warning asociadso a zero_point.get_zpt

    Returns
    -------
    df_data: pd.DataFrame
        Datos con la columna extra de parallax_corrected. En el caso de que exista la columna
        la modifica.
    """
    needed_columns = [
        "parallax",
        "phot_g_mean_mag",
        "nu_eff_used_in_astrometry",
        "nu_eff_used_in_astrometry",
        "pseudocolour",
        "ecl_lat",
        "astrometric_params_solved",
    ]
    if not np.isin(needed_columns, df_data.columns).all():
        df_data["parallax_corrected"] = np.nan
        return df_data

    zpt.load_tables()
    parallax = df_data["parallax"].values
    phot_g_mean_mag = df_data["phot_g_mean_mag"].values
    nueffused = df_data["nu_eff_used_in_astrometry"].values
    pseudocolour = df_data["pseudocolour"].values
    ecl_lat = df_data["ecl_lat"].values
    astrometric_params_solved = df_data["astrometric_params_solved"].values
    zpvals = zpt.get_zpt(
        phot_g_mean_mag,
        nueffused,
        pseudocolour,
        ecl_lat,
        astrometric_params_solved,
        _warnings=warnings,
    )
    df_data["parallax_corrected"] = parallax - zpvals
    return df_data


def get_radio(coords: Table, radio_scale: float, radius_type: Optional[str] = None) -> float:
    """
    Función que devulve el radio asociado a la búsqueda
    Parameters
    ----------
    radio_scale
    radius_type
    coords

    Returns
    -------

    """
    radius = None
    if radius_type is None:
        radius_type = "vision_fold_radius"

    if radius_type == "vision_fold_radius":
        radius = radio_scale * coords["ANGULAR_SIZE"][0] / 60
    if radius_type == "core_radius":
        radius = radio_scale * coords["CORE_RADIUS"][0]
    if radius_type == "half_light_radius":
        radius = radio_scale * coords["HALF_LIGHT_RADIUS"][0]

    if radius is None:
        raise ValueError(
            'El tipo de radio no es correcto, pruebe con "core_radius", '
            '"half_light_radius" o "vision_fold_radius"'
        )
    return radius
