import logging
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astropy.units import Unit
from astroquery.gaia import Gaia
from astroquery.heasarc import Heasarc
from astroquery.simbad import Simbad

HEASARC_COLUMNS = [
    "NAME",
    "RA",
    "DEC",
    "CORE_RADIUS",
    "HALF_LIGHT_RADIUS",
    "CENTRAL_CONCENTRATION",
]


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
    custom_simbad.TIMEOUT = timeout
    results = custom_simbad.query_object(name)
    if results is None:
        raise DidntFindObject(f"No se encontró el objeto {name} en Simbad.")
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


def get_cluster(
    name_cluster: str,
    r_scale: Optional[float] = None,
    rc_scale: Optional[float] = None,
    rhl_scale: Optional[float] = None,
) -> Table:
    """
    Función que devuelve los objetos de un cluster extraidos del catálogo GAIA DR3.
    La función extrae el Rc y las coordenadas del catálogo de GlobularCluster de Heasarc.

    Parameters
    ----------
    name_cluster: str
        Nombre del cluster
    r_scale: Optional[float], default None
        Scala del radio en unidiades del campo aparente del cluster.
    rc_scale: Optional[float], default None
        Scala del radio en unidiades de Rc (core radius). Por defecto toma 5 arcos de segundo.
    rhl_scale: Optional[float], default None
        Scala del radio en unidiades de Rhl (half light radius).
        Por defecto toma 5 arcos de segundo.

    Returns
    -------
    results: Table
        Tabla con los elementos del cluster en el radios rc_scale * Rc

    """
    result_cluster = get_object_from_heasarc(name_cluster)
    coords = get_skycoords(result_cluster, u.deg, u.deg)
    radius = 5 * u.arcsec

    ra = coords.ra.value  # Ascensión recta (RA) en grados
    dec = coords.dec.value  # Declinación (Dec) en grados

    if r_scale:
        radius = (
            np.power(10, result_cluster["CENTRAL_CONCENTRATION"].value[0])
            * result_cluster["CORE_RADIUS"].value[0]
            / 60
        )
    if rc_scale:
        radius = rc_scale * result_cluster["CORE_RADIUS"][0] * u.arcmin
    if rhl_scale:
        radius = rhl_scale * result_cluster["HALF_LIGHT_RADIUS"][0] * u.arcmin

    job = Gaia.launch_job_async(
        f"""
    SELECT * FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius * 1})
    )
    """,
        dump_to_file=True,
        output_format="votable",
        output_file="DR3_GCngc104_r.vot",
    )
    logging.info(job)
    results = job.get_results()
    results = results.to_pandas()
    logging.info(f"Se encontraron {len(results)} fuentes en el radio de búsqueda:")
    return results
