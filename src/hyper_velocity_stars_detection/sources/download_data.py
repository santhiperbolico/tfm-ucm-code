import logging

import astropy.units as u
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
